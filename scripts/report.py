"""Print the demo story straight from the deterministic tools (no LLM)."""
from risklens.tools import call_tool


def m(x):
    return f"{'-' if x < 0 else '+' if x > 0 else ' '}${abs(x) / 1e6:,.2f}M"


def main():
    a = call_tool("compare_allowance", {}).result
    print(f"Allowance {a['previous_date']}: ${a['previous_allowance'] / 1e6:,.2f}M -> {a['current_date']}: "
          f"${a['current_allowance'] / 1e6:,.2f}M  ({m(a['change'])}, {a['change_pct']:.1f}%)\n")
    rows = {meth: call_tool("calculate_allowance_attribution", {"method": meth}).result for meth in ("sequential", "reverse", "shapley")}
    print(f"{'Driver':<36}{'Sequential':>14}{'Reverse':>14}{'Shapley':>14}")
    for d in rows["sequential"]["drivers"]:
        vals = [next(x["impact"] for x in rows[k]["drivers"] if x["driver_id"] == d["driver_id"]) for k in rows]
        print(f"{d['driver']:<36}" + "".join(f"{m(v):>14}" for v in vals))
    r = call_tool("reconcile_allowance_change", {}).result
    print(f"\nReconciliation: residual ${r['residual']:,.2f} -> {r['status']} "
          f"(primary driver stable across methods: {r['robustness']['primary_driver_stable']})")
    seg = call_tool("calculate_segment_attribution", {}).result
    print("\nBy portfolio: " + ", ".join(f"{p} {m(seg['by_portfolio'][p]['total_change'])}" for p in seg["ranking"]))
    v = call_tool("calculate_variable_attribution", {}).result
    print("Macro by variable: " + ", ".join(f"{k} {m(v['by_variable'][k])}" for k in v["ranking"][:4]))
    top = call_tool("identify_top_contributors", {"n": 3}).result
    print("Top borrowers: " + "; ".join(f"{t['counterparty_id']} {m(t['ecl_change'])}" for t in top["rows"]))
    mv = call_tool("compare_model_versions", {}).result
    print(f"Model versions changed: {mv['any_model_changed']}")


if __name__ == "__main__":
    main()
