"""Allowance movement attribution that reconciles by construction, not by plugging.

A counterfactual state is defined by which input groups have moved from the
previous to the current period:

  runoff      exits, paydowns, amortisation and term changes on existing loans
  growth      new originations
  credit      borrower credit attributes (FICO, DTI, LTV, DSCR, rating, watchlist, DPD)
  macro       macroeconomic scenario paths
  methodology scenario weights, model versions and management overlays

The engine is re-run for each state, so

  * the sequential waterfall telescopes exactly to the reported change,
  * the reverse-order waterfall and the Shapley value (average over all
    orderings) show how sensitive the split is to the chosen order,
  * macro is split further by variable with an exact Shapley decomposition.

The residual against the booked (reported) allowance is measured, never forced.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import math
import pickle
from functools import lru_cache

import numpy as np
import pandas as pd

from risklens import config
from risklens.cecl.engine import RunInputs, run_cecl
from risklens.data.store import DataStore, get_store

GROUPS = ("runoff", "growth", "credit", "macro", "methodology")
MODEL_FIELDS = ["pd_model_id", "lgd_model_id", "ead_model_id"]


class Counterfactuals:
    """Engine evaluations for every combination of moved input groups (cached)."""

    def __init__(self, store: DataStore, prev: str, cur: str):
        self.store, self.prev, self.cur = store, prev, cur
        att = config.methodology()["attribution"]
        self.pf_fields = att["portfolio_fields"]
        self.cr_fields = att["credit_fields"]
        P = store.loans(prev).set_index("loan_id")
        C = store.loans(cur).set_index("loan_id")
        self.P, self.C = P, C
        self.survivors = P.index.intersection(C.index)
        self.exited = P.index.difference(C.index)
        self.new = C.index.difference(P.index)
        self.dev = {prev: store.macro_dev(prev), cur: store.macro_dev(cur)}
        self._cache: dict[frozenset, tuple[float, dict, pd.Series]] = {}
        self._macro_cache: dict[frozenset, tuple[float, dict]] = {}

    def _loans(self, moved: frozenset) -> pd.DataFrame:
        S = self.P.loc[self.survivors].copy()
        Cs = self.C.loc[self.survivors]
        if "runoff" in moved:
            S[self.pf_fields] = Cs[self.pf_fields]
        if "credit" in moved:
            S[self.cr_fields] = Cs[self.cr_fields]
        if "methodology" in moved:
            S[MODEL_FIELDS] = Cs[MODEL_FIELDS]
        parts = [S]
        if "runoff" not in moved:
            parts.append(self.P.loc[self.exited])
        if "growth" in moved:
            parts.append(self.C.loc[self.new])
        return pd.concat(parts).reset_index()

    def _inputs(self, moved: frozenset, macro_dev=None) -> RunInputs:
        side = self.cur if "methodology" in moved else self.prev
        dev = macro_dev if macro_dev is not None else self.dev[self.cur if "macro" in moved else self.prev]
        return RunInputs(f"counterfactual:{sorted(moved)}", self._loans(moved), dev,
                         self.store.weights(side), self.store.overlay_list(side))

    def evaluate(self, moved) -> tuple[float, dict, pd.Series]:
        """(allowance, allowance by portfolio, ECL by loan_id) with `moved` groups at current values."""
        key = frozenset(moved)
        if key not in self._cache:
            res = run_cecl(self._inputs(key))
            self._cache[key] = (res.allowance, res.by_portfolio(), pd.Series(res.ecl, index=res.loan_id))
        return self._cache[key]

    def evaluate_macro(self, base_moved, variables) -> tuple[float, dict]:
        """Allowance with `base_moved` groups moved and only `variables` on the current macro paths."""
        key = frozenset(variables)
        if key not in self._macro_cache:
            names = config.methodology()["macro"]["variables"]
            dev = self.dev[self.prev].copy()
            for v in variables:
                i = names.index(v)
                dev[:, i, :] = self.dev[self.cur][:, i, :]
            res = run_cecl(self._inputs(frozenset(base_moved), macro_dev=dev))
            self._macro_cache[key] = (res.allowance, res.by_portfolio())
        return self._macro_cache[key]


# ---------------------------------------------------------------------------
# Cache: counterfactual evaluations are deterministic functions of the data and
# the methodology, so they are stored on disk keyed by a hash of both.

def _fingerprint(store: DataStore, prev: str, cur: str) -> str:
    h = hashlib.sha256()
    for name in ("methodology", "models"):
        h.update(json.dumps(config.load(name), sort_keys=True, default=str).encode())
    for d in (prev, cur):
        h.update((store.dir / f"loans_{d}.parquet").read_bytes())
    h.update((store.dir / "macro_scenarios.csv").read_bytes())
    h.update((store.dir / "management_overlays.csv").read_bytes())
    return h.hexdigest()[:16]


@lru_cache(maxsize=8)
def _counterfactuals(data_dir: str, prev: str, cur: str) -> Counterfactuals:
    store = get_store(data_dir)
    cf = Counterfactuals(store, prev, cur)
    path = config.output_dir() / "cache" / f"cf_{prev}_{cur}_{_fingerprint(store, prev, cur)}.pkl"
    if path.exists():
        try:
            with open(path, "rb") as f:
                cf._cache, cf._macro_cache = pickle.load(f)
        except Exception:
            pass
    cf._cache_path = path
    return cf


def _save(cf: Counterfactuals) -> None:
    path = cf._cache_path
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump((cf._cache, cf._macro_cache), f)


def counterfactuals(prev: str, cur: str, store: DataStore | None = None) -> Counterfactuals:
    store = store or get_store()
    return _counterfactuals(str(store.dir), prev, cur)


# ---------------------------------------------------------------------------

def _driver_of(step_id: str) -> str:
    for s in config.methodology()["attribution"]["steps"]:
        if s["id"] == step_id:
            return s["driver"]
    raise KeyError(step_id)


def waterfall(prev: str, cur: str, order: tuple[str, ...] | None = None, store: DataStore | None = None) -> dict:
    """Sequential counterfactual waterfall; components sum exactly to the change."""
    cf = counterfactuals(prev, cur, store)
    steps = config.methodology()["attribution"]["steps"]
    order = tuple(order or [s["id"] for s in steps])
    labels = {s["id"]: s["label"] for s in steps}
    start, start_pf, _ = cf.evaluate(frozenset())
    moved: set[str] = set()
    level, level_pf = start, start_pf
    rows = []
    for g in order:
        moved.add(g)
        a, pf, _ = cf.evaluate(frozenset(moved))
        rows.append({"step": g, "driver": _driver_of(g), "label": labels[g], "impact": a - level,
                     "impact_by_portfolio": {p: pf[p] - level_pf[p] for p in pf}, "allowance_after": a})
        level, level_pf = a, pf
    _save(cf)
    return {"order": list(order), "start_allowance": start, "end_allowance": level, "steps": rows}


def shapley_groups(prev: str, cur: str, store: DataStore | None = None) -> dict[str, float]:
    """Exact Shapley value of each input group (average over all 120 orderings)."""
    cf = counterfactuals(prev, cur, store)
    n = len(GROUPS)
    vals = {g: 0.0 for g in GROUPS}
    for g in GROUPS:
        others = [x for x in GROUPS if x != g]
        for k in range(n):
            w = math.factorial(k) * math.factorial(n - k - 1) / math.factorial(n)
            for sub in itertools.combinations(others, k):
                with_g = cf.evaluate(frozenset(sub) | {g})[0]
                without = cf.evaluate(frozenset(sub))[0]
                vals[g] += w * (with_g - without)
    _save(cf)
    return vals


def driver_totals(step_rows: list[dict]) -> dict[str, float]:
    out: dict[str, float] = {}
    for r in step_rows:
        out[r["driver"]] = out.get(r["driver"], 0.0) + r["impact"]
    return out


def macro_variable_shapley(prev: str, cur: str, store: DataStore | None = None) -> dict:
    """Exact Shapley split of the macro step by variable, total and by portfolio.

    Base state = the sequential waterfall's state just before the macro step,
    so the variable contributions sum exactly to the macro component.
    """
    cf = counterfactuals(prev, cur, store)
    steps = [s["id"] for s in config.methodology()["attribution"]["steps"]]
    base = frozenset(steps[: steps.index("macro")])
    names = config.methodology()["macro"]["variables"]
    n = len(names)
    tot = {v: 0.0 for v in names}
    by_pf = {v: {} for v in names}
    for v in names:
        others = [x for x in names if x != v]
        for k in range(n):
            w = math.factorial(k) * math.factorial(n - k - 1) / math.factorial(n)
            for sub in itertools.combinations(others, k):
                a1, p1 = cf.evaluate_macro(base, set(sub) | {v})
                a0, p0 = cf.evaluate_macro(base, set(sub))
                tot[v] += w * (a1 - a0)
                for p in p1:
                    by_pf[v][p] = by_pf[v].get(p, 0.0) + w * (p1[p] - p0[p])
    _save(cf)
    return {"by_variable": tot, "by_variable_and_portfolio": by_pf,
            "macro_component": sum(tot.values()), "base_state": sorted(base)}


def loan_level_steps(prev: str, cur: str, store: DataStore | None = None) -> pd.DataFrame:
    """Per-loan ECL at each sequential step (index = loan_id union)."""
    cf = counterfactuals(prev, cur, store)
    steps = [s["id"] for s in config.methodology()["attribution"]["steps"]]
    cols = {"start": cf.evaluate(frozenset())[2]}
    moved: set[str] = set()
    for g in steps:
        moved.add(g)
        cols[g] = cf.evaluate(frozenset(moved))[2]
    df = pd.DataFrame(cols).fillna(0.0)
    prev_col = "start"
    out = pd.DataFrame(index=df.index)
    for g in steps:
        out[g] = df[g] - df[prev_col]
        prev_col = g
    out["ecl_prev"] = df["start"]
    out["ecl_cur"] = df[steps[-1]]
    return out
