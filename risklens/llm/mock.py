"""Deterministic keyword planner that speaks the same protocol as a real model.

Used by tests and for offline demos: turn 1 chooses tool calls from keywords
in the question; turn 2 writes the answer purely from the tool results it was
given (so it exercises the same grounding check as a real model).
"""
from __future__ import annotations

import json
import re

from risklens.llm.provider import ChatResponse, LLMProvider, ToolCall

P, C = "2026-07-31", "2026-08-31"
D = {"previous_date": P, "current_date": C}

PLANS = [
    (r"reconcil|tie out|add up", [("reconcile_allowance_change", D)]),
    (r"model|methodolog|calibrat|overlay", [("compare_model_versions", D), ("compare_management_overlays", D), ("compare_scenario_weights", D)]),
    (r"weight", [("compare_scenario_weights", D)]),
    (r"variable|unemploy|gdp|which macro", [("calculate_variable_attribution", D), ("compare_macro_scenarios", {**D, "scenario": "baseline"})]),
    (r"\bcre\b|commercial real estate", [("calculate_portfolio_attribution", {**D, "portfolio": "CRE"}),
                                          ("calculate_pd_migration", {**D, "portfolio": "CRE"}),
                                          ("identify_top_contributors", {**D, "portfolio": "CRE", "n": 5})]),
    (r"borrower|contributor|investigate|top \d+|which loans", [("identify_top_contributors", {**D, "n": 10})]),
    (r"macro|econom|scenario", [("calculate_macro_attribution", D), ("calculate_variable_attribution", D)]),
    (r"\bpd\b|probability of default", [("calculate_pd_migration", D)]),
    (r"\blgd\b|loss given", [("calculate_lgd_migration", D)]),
    (r"exposure|growth|origination|grow|balance|paydown|runoff", [("calculate_balance_change", D), ("calculate_new_originations", D)]),
    (r"which portfolio|segment|deteriorat", [("calculate_segment_attribution", D)]),
    (r"why|driver|cause|explain|increase|change", [("compare_allowance", D), ("calculate_allowance_attribution", D), ("reconcile_allowance_change", D)]),
    (r"july|august|allowance|how much", [("compare_allowance", D)]),
]


def m(x: float) -> str:
    return f"{'-' if x < 0 else ''}${abs(x) / 1e6:,.2f}M"


def _tool_results(messages: list[dict]) -> dict[str, dict]:
    out = {}
    for msg in messages:
        if msg.get("role") == "tool":
            try:
                body = json.loads(msg["content"])
            except (json.JSONDecodeError, TypeError):
                continue
            if body.get("ok"):
                out.setdefault(body["tool"], []).append(body["result"])
    return out


def _render(results: dict[str, list]) -> str:
    lines: list[str] = []
    if "compare_allowance" in results:
        r = results["compare_allowance"][0]
        lines.append(f"FACT: Allowance moved from {m(r['previous_allowance'])} on {r['previous_date']} to "
                     f"{m(r['current_allowance'])} on {r['current_date']}, a change of {m(r['change'])} ({r['change_pct']:.2f}%).")
    if "calculate_allowance_attribution" in results:
        r = results["calculate_allowance_attribution"][0]
        lines.append("FACT: Drivers of the change (" + r["method"] + " attribution):")
        for d in r["drivers"]:
            lines.append(f"- {d['driver']}: {m(d['impact'])} ({d['share_pct']:.1f}%)")
    for r in results.get("calculate_portfolio_attribution", []):
        lines.append(f"FACT: {r['portfolio']} allowance changed by {m(r['total_change'])}; primary driver: {r['primary_driver']}.")
        for s in r["steps"]:
            lines.append(f"- {s['driver']}: {m(s['impact'])}")
    if "calculate_segment_attribution" in results:
        r = results["calculate_segment_attribution"][0]
        lines.append(f"FACT: {r['largest_contributor']} contributed most ({m(r['by_portfolio'][r['largest_contributor']]['total_change'])}, "
                     f"{r['largest_contributor_share_pct']:.1f}% of the change).")
        for p in r["ranking"]:
            lines.append(f"- {p}: {m(r['by_portfolio'][p]['total_change'])}")
    if "calculate_macro_attribution" in results:
        r = results["calculate_macro_attribution"][0]
        lines.append(f"FACT: New macro scenario paths added {m(r['macro_paths_impact'])} ({r['share_of_total_change_pct']:.1f}% of the change); "
                     f"largest effect in {r['largest_portfolio']}.")
    for r in results.get("calculate_variable_attribution", []):
        lines.append(f"FACT: By macro variable ({r['portfolio']}), the top contributor is {r['top_variable']} at "
                     f"{m(r['by_variable'][r['top_variable']])} ({r['top_variable_share_pct']:.1f}% of the macro effect).")
        for v in r["ranking"]:
            lines.append(f"- {v}: {m(r['by_variable'][v])}")
    for r in results.get("compare_macro_scenarios", []):
        for s, vs in r["by_scenario"].items():
            u = vs["unemployment_rate"]
            lines.append(f"FACT: {s} unemployment (8-quarter average) moved from {u['previous']:.2f}% to {u['current']:.2f}%.")
    for r in results.get("calculate_pd_migration", []):
        a = r["all_loans"]
        lines.append(f"FACT: EAD-weighted 12-month PD ({r['portfolio']}) moved from {a['previous_pct']:.2f}% to {a['current_pct']:.2f}%; "
                     f"scorecard-only PD moved from {r['scorecard_pd_no_macro']['previous_pct']:.2f}% to {r['scorecard_pd_no_macro']['current_pct']:.2f}%.")
    for r in results.get("calculate_lgd_migration", []):
        a = r["all_loans"]
        lines.append(f"FACT: EAD-weighted LGD moved from {a['previous_pct']:.2f}% to {a['current_pct']:.2f}%.")
    for r in results.get("identify_top_contributors", []):
        lines.append(f"FACT: Top contributors by allowance increase ({r['portfolio']}):")
        for i, row in enumerate([x for x in r["rows"] if isinstance(x, dict)], 1):
            lines.append(f"{i}. {row['counterparty_id']} {row['customer_name']} ({row['portfolio']}): ECL {m(row['ecl_previous'])} -> "
                         f"{m(row['ecl_current'])} ({m(row['ecl_change'])}); PD {row['pd_12m_previous_pct']:.2f}% -> "
                         f"{row['pd_12m_current_pct']:.2f}%; " + "; ".join(row["reasons"]))
    if "calculate_balance_change" in results:
        r = results["calculate_balance_change"][0]
        lines.append(f"FACT: Balance moved from {m(r['previous_balance'])} to {m(r['current_balance'])}: exits {m(r['exits_paid_off_matured_charged_off'])}, "
                     f"existing loans {m(r['change_on_existing_loans'])}, new originations {m(r['new_originations'])}.")
    if "calculate_new_originations" in results:
        r = results["calculate_new_originations"][0]
        lines.append(f"FACT: New originations: {r['new_loans']} loans, {m(r['new_balance'])}; allowance impact {m(r['allowance_impact_portfolio_growth'])}.")
    if "compare_model_versions" in results:
        r = results["compare_model_versions"][0]
        cre = r["by_product"].get("Commercial Real Estate", {}).get("pd", {})
        lines.append("FACT: " + ("No PD, LGD or EAD model version changed between the periods." if not r["any_model_changed"]
                                 else "Model changes: " + ", ".join(r["changed"]) + "."))
        if cre:
            lines.append(f"FACT: CRE PD model: {cre['previous'][0]} in the previous period, {cre['current'][0]} in the current period.")
    if "compare_management_overlays" in results:
        r = results["compare_management_overlays"][0]
        lines.append(f"FACT: Management overlays: {m(r['previous_total'])} -> {m(r['current_total'])} (changed: {r['changed']}).")
    if "compare_scenario_weights" in results:
        r = results["compare_scenario_weights"][0]
        lines.append("FACT: Scenario weights " + ", ".join(f"{s} {r['previous'][s]:.0%} -> {r['current'][s]:.0%}" for s in r["previous"]) + ".")
    if "reconcile_allowance_change" in results:
        r = results["reconcile_allowance_change"][0]
        lines.append(f"FACT: Reported change {m(r['reported_change'])}; explained {m(r['explained_change'])}; residual "
                     f"${r['residual']:,.2f}; reconciliation {r['status']}. Primary driver stable across methods: "
                     f"{r['robustness']['primary_driver_stable']}.")
    if not lines:
        return "I could not answer this from the available tools."
    lines.append("INTERPRETATION: The figures above come from the deterministic tools; a risk professional should review them before acting.")
    return "\n".join(lines)


class MockProvider(LLMProvider):
    name = "mock"
    model = "mock-planner"

    def chat(self, messages, tools=None, tool_choice="auto") -> ChatResponse:
        results = _tool_results(messages)
        last_user = next((x["content"] for x in reversed(messages) if x["role"] == "user"), "")
        if results or not tools or tool_choice == "none":
            return ChatResponse(content=_render(results), model=self.model)
        q = last_user.lower()
        plan = []
        for pattern, calls in PLANS:
            if re.search(pattern, q):
                plan = calls
                break
        if not plan:
            plan = [("compare_allowance", D)]
        calls = [ToolCall(f"mock_{i}", name, json.dumps(args)) for i, (name, args) in enumerate(plan)]
        return ChatResponse(content="", tool_calls=calls, model=self.model)
