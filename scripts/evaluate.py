"""Evaluate the agent against the hidden ground truth on the 15 demo questions.

Scores tool selection (did it call a tool that can answer), success, grounding,
and whether the answer contains the expected fact. Works with any provider:

    python -m scripts.evaluate                    # mock planner (baseline for the pipeline)
    python -m scripts.evaluate --provider nvidia  # Nemotron via NVIDIA API
"""
import argparse
import json

from risklens import config
from risklens.agents.agent import InvestigatorAgent
from risklens.llm.provider import get_provider


def cases(gt):
    m = lambda x: f"{x / 1e6:.2f}"  # noqa: E731
    a = gt["allowance"]
    chg = a["2026-08-31"] - a["2026-07-31"]
    return [
        ("What was the July allowance?", {"compare_allowance", "get_allowance_summary"}, [m(a["2026-07-31"])]),
        ("What was the August allowance?", {"compare_allowance", "get_allowance_summary"}, [m(a["2026-08-31"])]),
        ("What was the dollar increase in allowance?", {"compare_allowance"}, [m(chg)]),
        ("By what percentage did the allowance increase?", {"compare_allowance"}, [f"{chg / a['2026-07-31'] * 100:.1f}"]),
        ("What were the major drivers of the allowance change?", {"calculate_allowance_attribution", "reconcile_allowance_change"}, ["Macroeconomic"]),
        ("Which portfolio contributed most to the change?", {"calculate_segment_attribution", "compare_allowance"}, [gt["primary_segment"]]),
        ("Which borrowers contributed most?", {"identify_top_contributors"}, [gt["largest_borrower"]]),
        ("Did PD increase?", {"calculate_pd_migration"}, ["PD"]),
        ("Did LGD increase?", {"calculate_lgd_migration"}, ["LGD"]),
        ("Did exposure grow?", {"calculate_balance_change", "calculate_ead_change", "calculate_new_originations"}, ["origination"]),
        ("Which macro variable had the largest impact?", {"calculate_variable_attribution"}, [gt["primary_macro_variable"].split("_")[0]]),
        ("Did scenario weights change?", {"compare_scenario_weights"}, ["12%"]),
        ("Did model versions change?", {"compare_model_versions"}, ["CRE_PD_v4"]),
        ("What happened specifically in CRE?", {"calculate_portfolio_attribution", "identify_top_contributors", "calculate_pd_migration"}, [gt["largest_borrower"]]),
        ("Does the attribution reconcile to the total allowance change?", {"reconcile_allowance_change"}, ["PASS"]),
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", choices=["mock", "nvidia", "vllm", "openai"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    with open(config.data_dir() / config.settings()["ground_truth_file"]) as f:
        gt = json.load(f)    # evaluation only: the agent never sees this
    agent = InvestigatorAgent(get_provider(args.provider))
    rows = []
    for q, expected_tools, must in cases(gt):
        run = agent.run(q)
        used = set(run.tools_used())
        rows.append({"question": q, "tool_selected": bool(used & expected_tools),
                     "answer_correct": all(s.lower() in run.answer.lower() for s in must),
                     "grounded": bool(run.grounding.get("ok")), **{k: run.metrics()[k] for k in
                     ("tool_calls", "tool_errors", "repaired_calls", "recovered_errors", "elapsed_s")}})
        r = rows[-1]
        print(f"{'OK ' if r['answer_correct'] else 'MISS'} tools={'Y' if r['tool_selected'] else 'N'} grounded={'Y' if r['grounded'] else 'N'}  {q}")
    n = len(rows)
    summary = {k: sum(r[k] for r in rows) / n for k in ("tool_selected", "answer_correct", "grounded")}
    summary["provider"] = agent.provider.name
    summary["model"] = agent.provider.model
    print(json.dumps(summary, indent=2))
    out = args.out or config.output_dir() / f"eval_{agent.provider.name}.json"
    with open(out, "w") as f:
        json.dump({"summary": summary, "rows": rows}, f, indent=2)


if __name__ == "__main__":
    main()
