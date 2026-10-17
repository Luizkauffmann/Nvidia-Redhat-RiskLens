"""Sequential counterfactual attribution of the allowance change.

Starting from the previous period's inputs, swap one input group at a time
to its current-period value and re-run the engine:

  previous -> exits -> balances/terms -> originations -> credit -> macro (per
  variable) -> scenario weights -> models -> overlays -> current

Each step's impact is the change in allowance it causes, so the steps
telescope to the total change. The final hybrid state is then compared to an
*independent* run on the actual current inputs: if an engine input changed
that no step swaps, it shows up as a residual. That is what the
Reconciliation Agent checks — the residual is not forced to zero.

Order matters for interaction effects; the order is part of the methodology
(config/methodology.yaml -> attribution.steps) and is reported with results.

This replaces the legacy Dataiku `build_attribution`, which rescaled the
components to sum to the total.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.cecl.engine import ECLEngine, EngineInputs, EngineResult
from src.data.loader import DataStore
from src.utils import config


@dataclass
class AttributionResult:
    summary: dict[str, Any]
    loan_contributions: pd.DataFrame          # one row per loan in either period
    portfolio_contributions: pd.DataFrame     # portfolio x step
    steps: list[dict[str, Any]] = field(default_factory=list)
    # Engine inputs just before the macro step (for exact macro Shapley).
    pre_macro: dict[str, Any] | None = field(default=None, repr=False)

    def to_dict(self) -> dict[str, Any]:
        return {**self.summary, "steps": self.steps,
                "by_portfolio": self.portfolio_contributions.reset_index().to_dict("records")}


def _hybrid_macro(prev: pd.DataFrame, curr: pd.DataFrame, swapped: list[str]) -> pd.DataFrame:
    key = ["scenario", "forecast_quarter"]
    out = prev.set_index(key).copy()
    if swapped:
        c = curr.set_index(key)
        if set(out.index) != set(c.index):
            raise ValueError("Macro scenario structure differs between periods; cannot swap variables")
        out[swapped] = c.loc[out.index, swapped]
    return out.reset_index()


class AllowanceAttributor:
    def __init__(self, store: DataStore | None = None, engine: ECLEngine | None = None):
        self.store = store or DataStore()
        self.engine = engine or ECLEngine()
        self.cfg = config.methodology()["attribution"]
        self.variables = config.methodology()["macro"]["variables"]

    # ------------------------------------------------------------------ #
    # State machinery shared by the sequential walk and the Shapley check.
    # Each step is a *switch* from previous to current inputs; the state
    # reached after a set of switches does not depend on their order.
    def _context(self, previous_date: str, current_date: str) -> dict[str, Any]:
        st, cfg = self.store, self.cfg
        P = st.loans_at(previous_date).set_index("loan_id", drop=False)
        C = st.loans_at(current_date).set_index("loan_id", drop=False)
        return {
            "previous_date": previous_date, "current_date": current_date, "P": P, "C": C,
            "cont": P.index.intersection(C.index),
            "new": C.index.difference(P.index, sort=False),
            "exits": P.index.difference(C.index, sort=False),
            "prev_assign": st.model_assignments_at(previous_date).set_index("product_type"),
            "macro_p": st.macro_at(previous_date), "macro_c": st.macro_at(current_date),
            "w_p": st.weights_at(previous_date), "w_c": st.weights_at(current_date),
            "ov_p": st.overlays_at(previous_date), "ov_c": st.overlays_at(current_date),
            "portfolio_fields": [f for f in cfg["portfolio_fields"] if f in C.columns],
            "credit_fields": [f for f in cfg["credit_fields"] if f in C.columns],
            "model_fields": [f for f in cfg["model_fields"] if f in C.columns],
        }

    @staticmethod
    def _initial_state(ctx: dict[str, Any]) -> dict[str, Any]:
        return {"loans": ctx["P"].copy(), "macro": ctx["macro_p"], "weights": ctx["w_p"], "overlays": ctx["ov_p"]}

    @staticmethod
    def _apply(ctx: dict[str, Any], state: dict[str, Any], sid: str, done: set[str]) -> None:
        """Switch one input group from previous to current (mutates ``state``)."""
        L, C = state["loans"], ctx["C"]
        cont = ctx["cont"]
        if sid == "runoff_exits":
            state["loans"] = L.loc[L.index.difference(ctx["exits"], sort=False)].copy()
        elif sid == "balance_term":
            L.loc[cont, ctx["portfolio_fields"]] = C.loc[cont, ctx["portfolio_fields"]]
        elif sid == "new_originations":
            add = C.loc[ctx["new"]].copy()
            if "model_change" not in done:   # models change only at the model step
                for f in ctx["model_fields"]:
                    add[f] = add["product_type"].map(ctx["prev_assign"][f])
            state["loans"] = pd.concat([L, add])
        elif sid == "credit_migration":
            L.loc[cont, ctx["credit_fields"]] = C.loc[cont, ctx["credit_fields"]]
        elif sid == "macro":
            state["macro"] = ctx["macro_c"]
        elif sid == "scenario_weights":
            state["weights"] = ctx["w_c"]
        elif sid == "model_change":
            idx = L.index.intersection(C.index)   # exited loans keep their models
            L.loc[idx, ctx["model_fields"]] = C.loc[idx, ctx["model_fields"]]
        elif sid == "overlay":
            state["overlays"] = ctx["ov_c"]
        else:
            raise ValueError(f"Unknown attribution step '{sid}'")

    def driver_shapley(self, previous_date: str, current_date: str, framework: str = "CECL") -> dict[str, Any]:
        """Order-independent check: exact Shapley values over driver groups.

        Each driver (paydowns/runoff, growth, credit, macro, model &
        judgment) is one player; its steps switch together. With D drivers
        this needs 2^D engine runs (32 for the default config). Interaction
        effects — e.g. credit deterioration hitting harder under a worse
        macro path — are split evenly instead of going to whichever step
        happens to come later in the sequential walk.
        """
        from itertools import combinations
        from math import factorial

        ctx = self._context(previous_date, current_date)
        order = [s["id"] for s in self.cfg["steps"]]
        driver_of = {s["id"]: s["driver"] for s in self.cfg["steps"]}
        D = list(dict.fromkeys(driver_of[s] for s in order))
        k = len(D)
        val: dict[frozenset, float] = {}
        port: dict[frozenset, pd.Series] = {}
        for r in range(k + 1):
            for combo in combinations(D, r):
                S = frozenset(combo)
                state, done = self._initial_state(ctx), set()
                for sid in order:
                    if driver_of[sid] in S:
                        self._apply(ctx, state, sid, done)
                        done.add(sid)
                res = self.engine.run(EngineInputs(as_of_date=current_date, loans=state["loans"].reset_index(drop=True),
                                                   macro=state["macro"], weights=state["weights"],
                                                   overlays=state["overlays"]), framework)
                val[S] = res.allowance
                by_p = res.loans.groupby("portfolio")["ecl"].sum()
                if len(state["overlays"]):
                    by_p = by_p.add(state["overlays"].groupby("portfolio")["amount"].sum(), fill_value=0.0)
                port[S] = by_p
        phi = {d: 0.0 for d in D}
        phi_port = {d: pd.Series(dtype=float) for d in D}
        for S in val:
            for d in D:
                if d in S:
                    continue
                w = factorial(len(S)) * factorial(k - len(S) - 1) / factorial(k)
                phi[d] += w * (val[S | {d}] - val[S])
                phi_port[d] = phi_port[d].add(w * port[S | {d}].sub(port[S], fill_value=0.0), fill_value=0.0)
        change = val[frozenset(D)] - val[frozenset()]
        labels = self.cfg["drivers"]
        return {
            "method": "exact_shapley_over_drivers",
            "engine_runs": len(val),
            "previous_allowance": float(val[frozenset()]),
            "current_allowance": float(val[frozenset(D)]),
            "change": float(change),
            "drivers": sorted(({"driver": d, "label": labels[d], "impact": float(phi[d]),
                                "share_of_change_pct": 100.0 * phi[d] / change if change else 0.0} for d in D),
                              key=lambda x: -abs(x["impact"])),
            "by_portfolio": pd.DataFrame(phi_port).fillna(0.0),
        }

    def run(self, previous_date: str, current_date: str, framework: str = "CECL",
            step_order: list[str] | None = None, macro_order: list[str] | None = None) -> AttributionResult:
        """Run the waterfall. ``step_order`` / ``macro_order`` override the
        configured order (used for order-sensitivity checks)."""
        st, eng, cfg = self.store, self.engine, self.cfg
        specs = {s["id"]: s for s in cfg["steps"]}
        order = list(step_order) if step_order else [s["id"] for s in cfg["steps"]]
        if sorted(order) != sorted(specs):
            raise ValueError(f"step_order must be a permutation of {sorted(specs)}")
        variables = list(macro_order) if macro_order else list(self.variables)
        if sorted(variables) != sorted(self.variables):
            raise ValueError(f"macro_order must be a permutation of {self.variables}")
        ctx = self._context(previous_date, current_date)
        P, C = ctx["P"], ctx["C"]
        cont, new, exits = ctx["cont"], ctx["new"], ctx["exits"]
        macro_p, macro_c = ctx["macro_p"], ctx["macro_c"]
        ov_p, ov_c = ctx["ov_p"], ctx["ov_c"]

        state = self._initial_state(ctx)
        done: set[str] = set()
        pre_macro: dict[str, Any] = {}

        def run_state() -> EngineResult:
            return eng.run(EngineInputs(as_of_date=current_date, loans=state["loans"].reset_index(drop=True),
                                        macro=state["macro"], weights=state["weights"], overlays=state["overlays"]),
                           framework)

        def ecl_series(res: EngineResult) -> pd.Series:
            return pd.Series(res.loans["ecl"].to_numpy(), index=res.loans["loan_id"].to_numpy())

        r_prev = run_state()
        snapshots: list[tuple[str, pd.Series, float]] = [("start", ecl_series(r_prev), r_prev.allowance)]
        step_meta: list[dict[str, Any]] = []
        labels = specs

        def record(step_id: str, sub: str | None = None, res: EngineResult | None = None):
            res = res or run_state()
            sid = f"{step_id}:{sub}" if sub else step_id
            snapshots.append((sid, ecl_series(res), res.allowance))
            impact = res.allowance - snapshots[-2][2]
            meta = labels[step_id]
            step_meta.append({"id": sid, "step": step_id, "driver": meta["driver"],
                              "label": f"{meta['label']} — {sub}" if sub else meta["label"],
                              "impact": float(impact), "allowance_after": float(res.allowance)})

        for sid in order:
            if sid == "macro":
                pre_macro = {"loans": state["loans"].copy(), "weights": state["weights"],
                             "overlays": state["overlays"]}
                swapped: list[str] = []
                for v in variables:
                    swapped.append(v)
                    state["macro"] = _hybrid_macro(macro_p, macro_c, swapped)
                    record(sid, sub=v)
            else:
                self._apply(ctx, state, sid, done)
                record(sid)
            done.add(sid)

        # independent run on actual current inputs
        r_curr = eng.run(st.engine_inputs(current_date), framework)
        prev_allow, curr_allow = r_prev.allowance, r_curr.allowance
        change = curr_allow - prev_allow
        explained = float(sum(s["impact"] for s in step_meta))
        residual = change - explained

        # ---- loan-level contributions --------------------------------- #
        all_ids = P.index.union(C.index, sort=False)
        mat = pd.DataFrame({sid: s.reindex(all_ids).fillna(0.0) for sid, s, _ in snapshots}, index=all_ids)
        deltas = mat.diff(axis=1).iloc[:, 1:]
        actual_curr = ecl_series(r_curr).reindex(all_ids).fillna(0.0)
        ref = pd.concat([P.loc[:, ["counterparty_id", "portfolio", "product_type"]],
                         C.loc[new, ["counterparty_id", "portfolio", "product_type"]]]).reindex(all_ids)
        loan_contrib = pd.concat([ref, pd.DataFrame({
            "status": np.where(all_ids.isin(new), "NEW", np.where(all_ids.isin(exits), "EXITED", "CONTINUING")),
            "ecl_previous": mat["start"], "ecl_current": actual_curr,
            "ecl_change": actual_curr - mat["start"]}, index=all_ids), deltas], axis=1)
        loan_contrib.index.name = "loan_id"

        # ---- portfolio x step (overlays allocated by their portfolio) ----- #
        by_port = deltas.groupby(loan_contrib["portfolio"]).sum()
        ov_delta = (ov_c.groupby("portfolio")["amount"].sum()
                    .sub(ov_p.groupby("portfolio")["amount"].sum(), fill_value=0.0))
        by_port["overlay"] = ov_delta.reindex(by_port.index).fillna(0.0)
        by_port["total"] = by_port.sum(axis=1)

        drivers_cfg = cfg["drivers"]
        drivers = []
        for d, label in drivers_cfg.items():
            imp = float(sum(s["impact"] for s in step_meta if s["driver"] == d))
            drivers.append({"driver": d, "label": label, "impact": imp,
                            "share_of_change_pct": 100.0 * imp / change if change else 0.0})
        drivers.sort(key=lambda x: -abs(x["impact"]))
        macro_vars = [{"variable": s["id"].split(":", 1)[1], "impact": s["impact"]}
                      for s in step_meta if s["step"] == "macro"]

        summary = {
            "framework": framework,
            "previous_date": previous_date,
            "current_date": current_date,
            "previous_allowance": float(prev_allow),
            "current_allowance": float(curr_allow),
            "change": float(change),
            "change_pct": 100.0 * change / prev_allow if prev_allow else None,
            "explained_change": explained,
            "residual": float(residual),
            "drivers": drivers,
            "macro_variables": sorted(macro_vars, key=lambda x: -abs(x["impact"])),
            "loan_counts": {"previous": int(len(P)), "current": int(len(C)), "continuing": int(len(cont)),
                            "exited": int(len(exits)), "new": int(len(new))},
            "method": {
                "type": "sequential_counterfactual",
                "step_order": order,
                "macro_variable_order": variables,
                "note": "Impacts depend on step order (interaction effects are assigned to the later step).",
            },
        }
        pre_macro.update(previous_date=previous_date, current_date=current_date, framework=framework)
        return AttributionResult(summary=summary, loan_contributions=loan_contrib,
                                 portfolio_contributions=by_port, steps=step_meta, pre_macro=pre_macro)

    # ------------------------------------------------------------------ #
    def macro_shapley(self, result: AttributionResult) -> dict[str, Any]:
        """Exact Shapley decomposition of the macro step across variables.

        Evaluates every subset of variables swapped to the current vintage
        (2^V engine projections on the state just before the macro step),
        then averages each variable's marginal contribution over all orders.
        Shapley values sum exactly to the macro step total and, unlike the
        sequential walk, do not depend on the variable order. Values are
        computed per loan, so portfolio splits add up too.
        """
        from itertools import combinations
        from math import factorial

        pm = result.pre_macro
        if not pm or "loans" not in pm:
            raise ValueError("Attribution result has no pre-macro state")
        st = self.store
        macro_p, macro_c = st.macro_at(pm["previous_date"]), st.macro_at(pm["current_date"])
        V = list(self.variables)
        k = len(V)
        loans = pm["loans"]
        inputs = EngineInputs(as_of_date=pm["current_date"], loans=loans.reset_index(drop=True),
                              macro=macro_p, weights=pm["weights"], overlays=pm["overlays"])
        subsets = [frozenset(c) for r in range(k + 1) for c in combinations(V, r)]
        frames = [_hybrid_macro(macro_p, macro_c, [v for v in V if v in S]) for S in subsets]
        evals = self.engine.macro_sweep(inputs, frames, pm["framework"])
        val = {S: e["loan_ecl"] for S, e in zip(subsets, evals)}
        tot = {S: e["allowance"] for S, e in zip(subsets, evals)}

        phi_loan = {v: np.zeros(len(loans)) for v in V}
        phi = {v: 0.0 for v in V}
        for S in subsets:
            for v in V:
                if v in S:
                    continue
                wgt = factorial(len(S)) * factorial(k - len(S) - 1) / factorial(k)
                Sv = S | {v}
                phi_loan[v] += wgt * (val[Sv] - val[S])
                phi[v] += wgt * (tot[Sv] - tot[S])
        portfolio = loans["portfolio"].to_numpy()
        by_port = (pd.DataFrame(phi_loan, index=loans.index).assign(portfolio=portfolio)
                   .groupby("portfolio").sum())
        total = tot[frozenset(V)] - tot[frozenset()]
        return {
            "method": "exact_shapley",
            "engine_runs": len(subsets),
            "macro_total": float(total),
            "variables": sorted(({"variable": v, "impact": float(phi[v])} for v in V), key=lambda x: -abs(x["impact"])),
            "by_portfolio": by_port,
            "loan_contributions": pd.DataFrame(phi_loan, index=loans.index),
        }
