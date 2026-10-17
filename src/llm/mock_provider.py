"""Deterministic stand-in for an LLM.

It plans tool calls from keywords in the question and writes the answer from
the tool results it receives, through exactly the same message protocol as a
real model. Uses: unit tests, CI, offline demos, and a "no-LLM" baseline for
the evaluation. It is clearly labelled and never used silently in place of
Nemotron.
"""
from __future__ import annotations

import json
import re
import uuid
from typing import Any

from src.llm.provider import LLMProvider, LLMResponse, ToolCall


def _m(x: float | None) -> str:
    return "n/a" if x is None else f"${x / 1e6:,.2f}M"


def _p(x: float | None, scale: float = 1.0) -> str:
    return "n/a" if x is None else f"{x * scale:.1f}%"


class MockProvider(LLMProvider):
    name = "mock"
    model = "mock-planner"

    def __init__(self, previous_date: str | None = None, current_date: str | None = None):
        self.previous_date, self.current_date = previous_date, current_date

    # ------------------------------------------------------------------ planning
    def _dates(self, messages: list[dict[str, Any]]) -> tuple[str, str]:
        if self.previous_date and self.current_date:
            return self.previous_date, self.current_date
        sys = messages[0]["content"] if messages and messages[0]["role"] == "system" else ""
        p = re.search(r"previous_date=(\d{4}-\d{2}-\d{2})", sys)
        c = re.search(r"current_date=(\d{4}-\d{2}-\d{2})", sys)
        return (p.group(1) if p else "2026-07-31"), (c.group(1) if c else "2026-08-31")

    def plan(self, question: str, prev: str, curr: str) -> list[tuple[str, dict[str, Any]]]:
        q = question.lower()
        pair = {"previous_date": prev, "current_date": curr}
        cre = "cre" in re.findall(r"[a-z&]+", q) or "commercial real estate" in q
        plan: list[tuple[str, dict[str, Any]]] = []
        factor = self._factor_scenario(q)
        if factor and any(w in q for w in ("what if", "what-if", "factor", "would", "scenario")):
            plan.append(("apply_scenario_factors", {"factor_scenario": factor, "as_of_date": curr}))
        if any(w in q for w in ("guidance", "regulat", "policy statement", "examiner", "audit", "supervis",
                                "sr 26", "sr 11", "require", "ecb", "esrb", "rules say")):
            plan.append(("search_regulation", {"query": question}))
        if "reconcil" in q or "add up" in q or "tie out" in q:
            plan.append(("reconcile_allowance_change", pair))
        if re.search(r"\bmodels?\b", q) and "macro" not in q:
            plan.append(("compare_model_versions", pair))
        if "weight" in q:
            plan.append(("compare_scenario_weights", pair))
        if any(w in q for w in ("macro", "unemployment", "variable", "economic", "gdp")):
            plan.append(("calculate_variable_attribution", pair))
        if cre:
            plan += [("calculate_segment_attribution", pair),
                     ("calculate_pd_migration", {**pair, "portfolio": "CRE"}),
                     ("identify_top_contributors", {**pair, "portfolio": "CRE", "n": 5})]
        elif any(w in q for w in ("borrower", "contributor", "investigate", "who ", "top ")):
            plan.append(("identify_top_contributors", {**pair, "n": 5}))
        if re.search(r"\bpd\b|probability of default", q) and not cre:
            plan.append(("calculate_pd_migration", pair))
        if re.search(r"\blgd\b|loss given default", q):
            plan.append(("calculate_lgd_migration", pair))
        if any(w in q for w in ("excel", "workbook", "spreadsheet", "report pack", "xlsx")):
            plan.append(("build_attribution_workbook", pair))
        if any(w in q for w in ("python", "pandas", "by state", "custom analysis", "write code")):
            plan.append(("describe_datasets", {}))
        if any(w in q for w in ("exposure", "growth", "balance", "origination", "grew")):
            plan.append(("calculate_balance_change", pair))
        if "which portfolio" in q or "portfolio contributed" in q:
            plan.append(("calculate_segment_attribution", pair))
        if not plan or (any(w in q for w in ("why", "driver", "what happened", "explain"))
                        and not any(n in ("search_regulation", "apply_scenario_factors") for n, _ in plan)):
            plan = [p for p in plan if p[0] != "describe_datasets"]
            plan = [("compare_allowance", pair), ("calculate_portfolio_attribution", pair)] + plan
        seen, out = set(), []
        for name, args in plan:
            k = (name, json.dumps(args, sort_keys=True))
            if k not in seen:
                seen.add(k)
                out.append((name, args))
        return out

    FACTOR_RULES = [("rates_plus_200bp", r"\b200\s*(bp|basis)|\+200bp"),
                    ("unemployment_7pct", r"(unemploy|jobless)\D{0,30}7\s*%|7\s*%\D{0,30}(unemploy|jobless)"),
                    ("cre_downturn_30pct", r"cre downturn|30\s*%\D{0,40}(commercial real estate|cre)|(commercial real estate|cre)\D{0,40}30\s*%"),
                    ("fed_2026_severely_adverse", r"severely adverse|stress test"),
                    ("baseline", r"\bbaseline factors?\b")]

    def _factor_scenario(self, q: str) -> str | None:
        return next((name for name, pat in self.FACTOR_RULES if re.search(pat, q)), None)

    # ------------------------------------------------------------------ protocol
    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None,
             **kwargs: Any) -> LLMResponse:
        question = next((m["content"] for m in messages if m["role"] == "user"), "")
        prev, curr = self._dates(messages)
        done = {}
        for m in messages:
            if m["role"] == "assistant" and m.get("tool_calls"):
                for tc in m["tool_calls"]:
                    done[tc["id"]] = tc["function"]["name"]
        results: dict[str, Any] = {}
        for m in messages:
            if m["role"] == "tool" and m["tool_call_id"] in done:
                payload = json.loads(m["content"])
                if "result" in payload:
                    results[done[m["tool_call_id"]]] = payload["result"]
        allowed = {t["function"]["name"] for t in (tools or [])}
        if any(n.startswith("ask_") for n in allowed):
            return self._team_turn(question, prev, curr, done, results, allowed, messages)
        pending = [(n, a) for n, a in self.plan(question, prev, curr) if n not in done.values() and n in allowed]
        if pending and tools:
            name, args = pending[0]
            return LLMResponse(tool_calls=[ToolCall(id=f"mock_{uuid.uuid4().hex[:8]}", name=name, arguments=args)],
                               raw_tool_format="native", finish_reason="tool_calls")
        return LLMResponse(content=self.compose(question, results), finish_reason="stop")

    # ------------------------------------------------------------------ team mode (orchestrator with ask_* tools)
    SPECIALIST_PRIORITY = ["reconciliation_agent", "reporting_agent", "quant_agent", "regulatory_agent", "model_risk_agent",
                           "macro_agent", "credit_agent", "portfolio_agent"]

    def _owner(self, tool_name: str, allowed: set[str]) -> str | None:
        from src.tools import list_tools
        for agent in self.SPECIALIST_PRIORITY:
            if f"ask_{agent}" in allowed and tool_name in list_tools(agent):
                return agent
        return None

    def _team_turn(self, question, prev, curr, done, results, allowed, messages) -> LLMResponse:
        plan = self.plan(question, prev, curr)
        targets: list[str] = []
        for name, _ in plan:
            if name in allowed:
                continue                         # orchestrator calls shared tools itself
            owner = self._owner(name, allowed)
            if owner and owner not in targets:
                targets.append(owner)
        direct = [(n, a) for n, a in plan if n in allowed and n not in done.values()][:1]
        pending = [t for t in targets if f"ask_{t}" not in done.values()]
        if direct:
            name, args = direct[0]
            return LLMResponse(tool_calls=[ToolCall(id=f"mock_{uuid.uuid4().hex[:8]}", name=name, arguments=args)],
                               raw_tool_format="native", finish_reason="tool_calls")
        if pending:
            return LLMResponse(tool_calls=[ToolCall(id=f"mock_{uuid.uuid4().hex[:8]}", name=f"ask_{pending[0]}",
                                                    arguments={"question": question})],
                               raw_tool_format="native", finish_reason="tool_calls")
        own = {k: v for k, v in results.items() if not k.startswith("ask_")}
        text = self.compose(question, own) if own else "Here is what the specialists found."
        head, _, rest = text.partition("\n")
        lines = [head]
        for k, v in results.items():
            if k.startswith("ask_"):
                lines += [f"**{k[4:].replace('_', ' ').title()}**", v.get("answer", "")]
        if rest:
            lines.append(rest)
        return LLMResponse(content="\n".join(lines), finish_reason="stop")

    # ------------------------------------------------------------------ answer writing
    def compose(self, question: str, r: dict[str, Any]) -> str:
        facts: list[str] = []
        head = "Here is what the tools show."
        if "compare_allowance" in r:
            c = r["compare_allowance"]
            head = (f"The {c['framework']} allowance {'increased' if c['change'] > 0 else 'decreased'} by "
                    f"{_m(abs(c['change']))} ({_p(c['change_pct'])}), from {_m(c['previous_allowance'])} to {_m(c['current_allowance'])}.")
            facts.append(f"Allowance {_m(c['previous_allowance'])} -> {_m(c['current_allowance'])}, change {_m(c['change'])} ({_p(c['change_pct'])}) [compare_allowance]")
        if "calculate_portfolio_attribution" in r:
            a = r["calculate_portfolio_attribution"]
            for d in a["drivers"][:5]:
                facts.append(f"{d['label']}: {_m(d['impact'])} ({_p(d['share_of_change_pct'])} of the change) [calculate_portfolio_attribution]")
            facts.append(f"Residual {_m(a['residual'])} [calculate_portfolio_attribution]")
        if "calculate_segment_attribution" in r:
            s = r["calculate_segment_attribution"]
            for seg in s["segments"][:3]:
                if "segment" in seg:
                    facts.append(f"{seg['segment']}: {_m(seg['total_change'])} ({_p(seg['share_of_total_change_pct'])} of the change), largest driver {seg['largest_driver']} [calculate_segment_attribution]")
        if "calculate_pd_migration" in r:
            t = r["calculate_pd_migration"]["total"]
            facts.append(f"12-month PD {_p(t['pd_12m_previous'], 100)} -> {_p(t['pd_12m_current'], 100)}; base scorecard PD {_p(t['base_pd_previous'], 100)} -> {_p(t['base_pd_current'], 100)} [calculate_pd_migration]")
        if "calculate_lgd_migration" in r:
            t = r["calculate_lgd_migration"]["total"]
            facts.append(f"LGD {_p(t['lgd_previous'], 100)} -> {_p(t['lgd_current'], 100)} [calculate_lgd_migration]")
        if "calculate_balance_change" in r:
            t = r["calculate_balance_change"]["total"]
            facts.append(f"Balance {_m(t['previous_balance'])} -> {_m(t['current_balance'])}; new originations {_m(t['new_originations'])} [calculate_balance_change]")
        if "identify_top_contributors" in r:
            for c in r["identify_top_contributors"]["contributors"][:5]:
                if "rank" not in c:
                    continue
                codes = ", ".join(c["reason_codes"][:3])
                facts.append(f"#{c['rank']} {c['counterparty_id']} ({c['customer_name']}, {c['portfolio']}): ECL {_m(c['ecl_change'])}, "
                             f"PD {_p(c['pd_12m_previous'], 100)} -> {_p(c['pd_12m_current'], 100)}; {codes} [identify_top_contributors]")
        if "calculate_variable_attribution" in r:
            v = r["calculate_variable_attribution"]
            for x in v["variables"][:3]:
                facts.append(f"{x['variable']}: {_m(x['impact'])} ({_p(x['share_of_macro_pct'])} of the macro effect) [calculate_variable_attribution]")
        if "compare_model_versions" in r:
            mv = r["compare_model_versions"]
            cands = ", ".join(m["model_id"] for m in mv["candidate_models_not_in_production"]) or "none"
            facts.append(f"Model versions changed: {'yes' if mv['any_model_version_changed'] else 'no'}; "
                         f"candidate models not in production: {cands} [compare_model_versions]")
        if "compare_scenario_weights" in r:
            w = r["compare_scenario_weights"]
            facts.append(f"Scenario weights changed: {'yes' if w['weights_changed'] else 'no'}"
                         + (f", shifted toward {w['weight_shifted_toward']}, impact {_m(w['allowance_impact_of_weight_change'])}" if w["weights_changed"] else "")
                         + " [compare_scenario_weights]")
        if "build_attribution_workbook" in r:
            b = r["build_attribution_workbook"]
            h = b["headline"]
            facts.append(f"Excel pack written to {b['path']} with sheets {', '.join(b['sheets'])}; change {_m(h['change'])}, "
                         f"residual {_m(h['residual'])}, reconciliation {h['reconciliation']} [build_attribution_workbook]")
        if "describe_datasets" in r:
            facts.append("Datasets available for custom analysis: " + ", ".join(t["table"] for t in r["describe_datasets"]["tables"])
                         + " [describe_datasets]. (mock planner: no code is written without a model)")
        if "apply_scenario_factors" in r:
            x = r["apply_scenario_factors"]
            head = (f"Under the {x['factor_scenario']} factor scenario the {x['framework']} allowance would be "
                    f"{_m(x['scenario_allowance'])} versus {_m(x['official_allowance'])} official ({_p(x['change_pct'])}).")
            for p in [p for p in x["by_portfolio"] if p["covered"]][:3]:
                facts.append(f"{p['portfolio']}: {_m(p['change'])} (PD x{p['pd_mult']}, LGD {p['lgd_adj']:+.2f}) [apply_scenario_factors]")
            if x["not_covered"]:
                facts.append(f"No factors on file for: {', '.join(x['not_covered'])} (left unchanged) [apply_scenario_factors]")
            facts.append(f"Method: {x['method']} Factors are illustrative team assumptions [apply_scenario_factors]")
        if "search_regulation" in r:
            hits = r["search_regulation"]["passages"]
            head = f"Here is what the guidance says, from {hits[0]['document']}." if hits else head
            for h in hits[:2]:
                snippet = re.split(r"(?<=[.;])\s", h["text"], maxsplit=1)[0][:240]
                facts.append(f"{snippet} [{h['citation']}] [search_regulation]")
        if "reconcile_allowance_change" in r:
            x = r["reconcile_allowance_change"]
            head = f"Reconciliation {x['overall_status']}: the attribution explains {_p(x['pct_explained'])} of the reported change."
            facts.append(f"Reported {_m(x['reported_change'])}, explained {_m(x['explained_change'])}, residual {_m(x['residual'])} [reconcile_allowance_change]")
        lines = [head, "**Facts**", *[f"- {f}" for f in facts], "**Interpretation**",
                 "- (mock planner) Template summary of the tool results above; no model reasoning involved."]
        return "\n".join(lines)
