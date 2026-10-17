"""Evaluation task bank: the demo questions plus what a correct answer must contain.

Expected values come from ``ground_truth.json`` (the story we injected and the
observed allowance figures), so this module is evaluation-only, like the ground
truth itself. Tools and agents never import it.

A task is a question plus:

* ``checks`` — what the final answer must say. Types:
    - ``mentions``      the answer matches at least one regex in ``any_of``
    - ``not_mentions``  the answer matches none of the regexes in ``none_of``
    - ``number``        some money/percent figure in the answer is within tolerance of ``value``
* ``required_tools`` — list of any-of groups; each group needs one of its tools called.
* ``call_budget`` — tool calls allowed before the efficiency score starts to drop.

The same task definitions feed the NeMo Gym dataset (``scripts/build_gym_dataset.py``)
and the in-process evaluator (``src/evaluation/local_eval.py``).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from src.utils import config

NEG = r"(?:\bno\b|\bnot\b|n't\b|\bunchanged\b|\bsame\b|\bnone\b|\bwithout\b)"
UP = r"(?:increas|rose|\bup\b|grew|higher|rise|rising|worsen|deteriorat)"


@dataclass
class Task:
    task_id: str
    question: str
    category: str
    checks: list[dict[str, Any]]
    required_tools: list[list[str]] = field(default_factory=list)
    call_budget: int = 6
    paraphrase_of: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_ground_truth(path: Path | str | None = None) -> dict[str, Any]:
    path = Path(path) if path else config.path("ground_truth") / "ground_truth.json"
    with Path(path).open() as fh:
        return json.load(fh)


def _money(value: float, desc: str, rel_tol: float | None = None) -> dict[str, Any]:
    c = {"type": "number", "kind": "money", "value": float(value), "desc": desc}
    if rel_tol is not None:
        c["rel_tol"] = rel_tol      # wider when sequential and Shapley attributions legitimately differ
    return c


def _mention(desc: str, *patterns: str) -> dict[str, Any]:
    return {"type": "mentions", "any_of": list(patterns), "desc": desc}


def _absent(desc: str, *patterns: str) -> dict[str, Any]:
    return {"type": "not_mentions", "none_of": list(patterns), "desc": desc}


DRIVER_PATTERNS = {
    "macroeconomic": r"macro|economic scenario|economic forecast",
    "credit_deterioration": r"credit deterioration|credit quality|deteriorat",
    "portfolio_growth": r"portfolio growth|new originations|growth",
    "model_methodology": r"model|methodolog|scenario weight",
    "paydowns_runoff": r"paydown|runoff|run-off",
}
VARIABLE_PATTERNS = {
    "unemployment_rate": r"unemployment",
    "cre_price_growth": r"cre price|commercial real estate price|property price|cre_price",
    "gdp_growth": r"\bgdp\b",
    "interest_rate": r"interest rate",
    "hpi_growth": r"\bhpi\b|house price|home price",
    "inflation": r"inflation",
}
PORTFOLIO_PATTERNS = {"CRE": r"\bCRE\b|commercial real estate", "C&I": r"C&I|commercial (?:and|&) industrial"}


def build_tasks(gt: dict[str, Any] | None = None, include_paraphrases: bool = True) -> list[Task]:
    """The 15 demo-success questions (README / project brief) with checks derived from ground truth."""
    gt = gt or load_ground_truth()
    d, obs = gt["design"], gt["observed_cecl"]
    prev, curr, change = obs["previous_allowance"], obs["current_allowance"], obs["change"]
    drivers = sorted(obs["drivers"].items(), key=lambda kv: -abs(kv[1]))
    primary_driver = d["primary_driver"]
    top_portfolio = max(obs["portfolio_change"], key=obs["portfolio_change"].get)
    top_var = max(obs["macro_variables"], key=obs["macro_variables"].get)
    borrower = d["largest_borrower"]
    pct = 100.0 * change / prev
    # story variants can hold the macro paths flat: then every variable's impact is exactly zero
    macro_moved = max(abs(v) for v in obs["macro_variables"].values()) > 1e-6 * abs(change)
    port_pat = PORTFOLIO_PATTERNS.get(top_portfolio, rf"\b{top_portfolio}\b")

    T: list[Task] = [
        Task("q01_previous_allowance", "What was the CECL allowance in July 2026?", "allowance",
             [_money(prev, "July allowance")], [["get_allowance_summary", "compare_allowance"]], 3),
        Task("q02_current_allowance", "What was the CECL allowance at the end of August 2026?", "allowance",
             [_money(curr, "August allowance")], [["get_allowance_summary", "compare_allowance"]], 3),
        Task("q03_dollar_change", "By how many dollars did the allowance change from July to August?", "allowance",
             [_money(change, "dollar change"), _mention("direction: increase", UP)], [["compare_allowance"]], 3),
        Task("q04_pct_change", "What was the percentage increase in the allowance between July and August?", "allowance",
             [{"type": "number", "kind": "percent", "value": pct, "desc": "percent change"}], [["compare_allowance"]], 3),
        Task("q05_drivers", "Why did our allowance increase in August? What were the major drivers?", "drivers",
             [_mention("names the primary driver", DRIVER_PATTERNS[primary_driver]),
              _mention("names the second driver", DRIVER_PATTERNS[drivers[1][0]]),
              _money(change, "total change"),
              _money(obs["drivers"][primary_driver], "primary driver impact")],
             [["calculate_portfolio_attribution"]], 6),
        Task("q06_top_portfolio", "Which portfolio contributed most to the allowance increase?", "portfolio",
             [_mention(f"names {top_portfolio}", port_pat), _money(obs["portfolio_change"][top_portfolio], "portfolio change")],
             [["calculate_segment_attribution", "compare_allowance"]], 4),
        Task("q07_top_borrowers", "Which borrowers contributed most to the increase? Who should I investigate first?", "borrowers",
             [_mention("names the largest borrower", rf"\b{borrower}\b")], [["identify_top_contributors"]], 4),
        Task("q08_pd", "Did probability of default increase between July and August?", "credit",
             [_mention("mentions PD", r"\bPD\b|probability of default"), _mention("says it increased", UP),
              _absent("does not deny the increase", r"\bPDs? (?:did not|didn't) (?:increase|rise)")],
             [["calculate_pd_migration"]], 4),
        Task("q09_lgd", "Did LGD increase, and did it drive the change?", "credit",
             [_mention("mentions LGD", r"\bLGD\b|loss given default")], [["calculate_lgd_migration"]], 4),
        Task("q10_exposure", "Did exposure grow? How much new business did we originate?", "portfolio",
             [_mention("mentions originations", r"originat|new (?:business|loans|exposure)"),
              _mention("mentions balance or exposure", r"balance|exposure|EAD")],
             [["calculate_balance_change", "calculate_new_originations", "compare_portfolios"]], 4),
        Task("q11_macro_variable", "Which macroeconomic variable had the largest impact on the allowance?", "macro",
             ([_mention(f"names {top_var}", VARIABLE_PATTERNS[top_var]),
               _money(obs["macro_variables"][top_var], "top variable impact", rel_tol=0.06)] if macro_moved else
              [_mention("says no macro variable moved the allowance",
                        rf"{NEG}|\bzero\b|negligible|\$0(?:\.0+)?(?:\s*M| million)?\b")]),
             [["calculate_variable_attribution"]], 4),
        Task("q12_weights", "Did the scenario weights change between July and August?", "model_risk",
             [_mention("says weights changed" if d["scenario_weights_changed"] else "says weights unchanged",
                       r"\b(?:yes|changed|shift|moved|increased|raised)\b" if d["scenario_weights_changed"] else NEG),
              *([_mention("mentions the severe/adverse scenario", r"severe|adverse|downside")]
                if d["scenario_weights_changed"] else [])],
             [["compare_scenario_weights"]], 3),
        Task("q13_model_change", "Could this increase just be because we changed models?", "model_risk",
             ([_mention("says no model version changed", rf"{NEG}[^.\n]{{0,80}}model|model[^.\n]{{0,80}}{NEG}"),
               _absent("does not claim a model change", r"model versions? changed: yes", r"\bnew model (?:was|has been) deployed")]
              if not d["model_changed"] else [_mention("says a model changed", r"model[^.\n]{0,60}changed")]),
             [["compare_model_versions"]], 3),
        Task("q14_cre", "What happened specifically in Commercial Real Estate?", "portfolio",
             [_mention("mentions CRE", PORTFOLIO_PATTERNS["CRE"]),
              _mention("mentions borrower deterioration", r"deteriorat|downgrad|DSCR|LTV|PD|rating"),
              _mention("names a deteriorated CRE borrower", *[rf"\b{c}\b" for c in d["deteriorated_cre_borrowers"][:10]])],
             [["calculate_segment_attribution", "identify_top_contributors", "calculate_pd_migration"]], 5),
        Task("q15_reconcile", "Does the attribution reconcile to the total allowance change?", "reconciliation",
             [_mention("reports PASS", r"\bPASS\b|reconciles|fully explained|100(?:\.0+)?%"),
              _money(change, "reported change")],
             [["reconcile_allowance_change"]], 3),
    ]

    if include_paraphrases:
        para = {
            "q05_drivers": ["What drove the CECL increase this month?", "Explain the July to August allowance movement."],
            "q06_top_portfolio": ["Which segment drove most of the allowance change?"],
            "q07_top_borrowers": ["Show me the top contributors to the ECL increase."],
            "q11_macro_variable": ["Which economic variable mattered most for the macro effect?"],
            "q13_model_change": ["Did we deploy a new PD or LGD model between July and August?"],
            "q15_reconcile": ["Does everything tie out?"],
            "q03_dollar_change": ["How much did the reserve go up from July to August, in dollars?"],
        }
        base = {t.task_id: t for t in T}
        for tid, qs in para.items():
            for i, q in enumerate(qs, start=1):
                b = base[tid]
                T.append(Task(f"{tid}_p{i}", q, b.category, b.checks, b.required_tools, b.call_budget, paraphrase_of=tid))
    return T
