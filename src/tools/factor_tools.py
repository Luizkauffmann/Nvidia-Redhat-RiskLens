"""Scenario factor tools: a first-order what-if on top of the official allowance.

A factor scenario (e.g. "unemployment rises to 7%") holds, per portfolio, a PD multiplier, an additive
LGD adjustment and an optional CCF for undrawn amounts (data/kb/scenario_factors.csv, loaded into the
knowledge base by `python -m scripts.load_kb`). The factors are ILLUSTRATIVE team assumptions.

apply_scenario_factors scales each portfolio's official modeled ECL deterministically:

    scenario ECL = modeled ECL x min(PD multiplier, 1 / EAD-weighted PD) x (LGD + adjustment) / LGD

Overlays are carried unchanged. This is a sensitivity, not a re-run of the loan-level engine; for the
engine's own scenario results use run_scenario_analysis.
"""
from __future__ import annotations

from typing import Any, Literal

from src.kb import store
from src.tools.allowance_tools import get_allowance_summary
from src.tools.base import Framework, ToolError, tool

FACTOR_AGENTS = ["orchestrator", "macro_agent"]
# Must match the scenarios in data/kb/scenario_factors.csv (tests/test_kb_tools.py checks this).
FactorScenario = Literal["baseline", "rates_plus_200bp", "unemployment_7pct", "cre_downturn_30pct",
                         "fed_2026_severely_adverse"]


def _factors(factor_scenario: str | None) -> list[dict[str, Any]]:
    try:
        return store.factors(factor_scenario)
    except store.KBUnavailable as exc:
        raise ToolError(str(exc), "dependency_error",
                        {"fix": "start Postgres, then run `python -m scripts.load_kb`"}) from exc


@tool("macro", FACTOR_AGENTS)
def get_scenario_factors(factor_scenario: FactorScenario | None = None) -> dict[str, Any]:
    """Scenario factors by portfolio (PD multiplier, LGD adjustment, CCF) for one factor scenario, or all of them; illustrative team assumptions."""
    rows = _factors(factor_scenario)
    if not rows:
        raise ToolError(f"No factors on file for {factor_scenario!r}", hint={"allowed": list(FactorScenario.__args__)})
    return {"factor_scenario": factor_scenario, "scenarios": sorted({r["scenario"] for r in rows}),
            "factors": [{k: r[k] for k in ("scenario", "portfolio", "pd_mult", "lgd_adj", "ead_ccf")} for r in rows],
            "source": rows[0]["source"]}


@tool("macro", FACTOR_AGENTS)
def apply_scenario_factors(factor_scenario: FactorScenario, as_of_date: str, framework: Framework = "CECL") -> dict[str, Any]:
    """What-if allowance if a factor scenario's PD/LGD factors were applied to the official run on as_of_date, by portfolio; a first-order sensitivity, not an engine re-run."""
    by_portfolio = {f["portfolio"]: f for f in _factors(factor_scenario)}
    if not by_portfolio:
        raise ToolError(f"No factors on file for {factor_scenario!r}", hint={"allowed": list(FactorScenario.__args__)})
    official = get_allowance_summary(as_of_date, framework)
    rows, not_covered = [], []
    for p in official["by_portfolio"]:
        f = by_portfolio.get(p["portfolio"])
        base_ecl, overlay = p["modeled_ecl"], p["overlay"]
        if f is None:
            not_covered.append(p["portfolio"])
            rows.append({"portfolio": p["portfolio"], "covered": False, "modeled_ecl": base_ecl, "scenario_modeled_ecl": base_ecl,
                         "overlay": overlay, "allowance": p["allowance"], "scenario_allowance": p["allowance"], "change": 0.0})
            continue
        pd0, lgd0 = p["ead_weighted_pd_12m"], p["ead_weighted_lgd"]
        pd_mult = min(f["pd_mult"], 1.0 / pd0) if pd0 > 0 else f["pd_mult"]       # stressed PD cannot exceed 100%
        lgd1 = min(1.0, max(0.0, lgd0 + f["lgd_adj"]))
        lgd_ratio = lgd1 / lgd0 if lgd0 > 0 else 1.0
        scen_ecl = base_ecl * pd_mult * lgd_ratio
        rows.append({"portfolio": p["portfolio"], "covered": True, "pd_mult": f["pd_mult"], "pd_mult_applied": pd_mult,
                     "lgd_adj": f["lgd_adj"], "lgd": lgd0, "lgd_scenario": lgd1,
                     "ccf_override": f["ead_ccf"], "ccf_applied": False,
                     "modeled_ecl": base_ecl, "scenario_modeled_ecl": scen_ecl, "overlay": overlay,
                     "allowance": p["allowance"], "scenario_allowance": scen_ecl + overlay,
                     "change": scen_ecl - base_ecl})
    base_total = official["allowance"]
    scen_total = sum(r["scenario_allowance"] for r in rows)
    return {"factor_scenario": factor_scenario, "as_of_date": as_of_date, "framework": framework,
            "official_allowance": base_total, "scenario_allowance": scen_total, "change": scen_total - base_total,
            "change_pct": 100.0 * (scen_total - base_total) / base_total if base_total else 0.0,
            "by_portfolio": sorted(rows, key=lambda r: -abs(r["change"])), "not_covered": not_covered,
            "method": "First-order sensitivity: modeled ECL x PD multiplier x (LGD + adjustment)/LGD per portfolio; "
                      "overlays unchanged; CCF overrides not applied (portfolio summary has no undrawn split).",
            "factor_source": next(iter(by_portfolio.values()))["source"]}
