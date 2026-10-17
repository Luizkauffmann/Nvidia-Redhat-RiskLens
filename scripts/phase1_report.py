"""Print the Phase 1 summary (dimensions, allowance, drivers) from generated data.

    python -m scripts.phase1_report
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.cecl.attribution import AllowanceAttributor  # noqa: E402
from src.data.loader import DataStore  # noqa: E402
from src.utils import config  # noqa: E402


def m(x: float) -> str:
    return f"${x / 1e6:,.2f}M"


def main() -> None:
    st = DataStore()
    d0, d1 = config.settings()["reporting_dates"]
    print("== Dataset dimensions")
    for name in ["loans", "counterparties", "macro", "weights", "overlays", "model_metadata",
                 "model_assignments", "model_parameters", "results", "summary"]:
        df = getattr(st, name)
        print(f"  {name:18s} {len(df):>8,} rows x {df.shape[1]:>2} cols")
    for d in (d0, d1):
        print(f"  loans @ {d}: {len(st.loans_at(d)):,}")

    print("\n== Allowance by framework")
    s = st.summary
    for _, r in s.iterrows():
        print(f"  {r['run_id']:26s} balance {m(r['total_balance']):>12}  allowance {m(r['allowance']):>10}  ({r['allowance_rate_pct']:.2f}%)")

    out = {}
    for fw in config.settings()["frameworks"]:
        a = AllowanceAttributor(st).run(d0, d1, fw)
        sm = a.summary
        out[fw] = a.to_dict()
        print(f"\n== {fw} attribution {d0} -> {d1}")
        print(f"  {m(sm['previous_allowance'])} -> {m(sm['current_allowance'])}  change {m(sm['change'])} ({sm['change_pct']:.1f}%)")
        for d in sm["drivers"]:
            print(f"    {d['label']:34s} {m(d['impact']):>10}  {d['share_of_change_pct']:6.1f}%")
        print(f"    {'Residual':34s} {m(sm['residual']):>10}")
        print("  Macro variables:", ", ".join(f"{x['variable']} {m(x['impact'])}" for x in sm["macro_variables"][:4]))
        print("  By portfolio:   ", ", ".join(f"{k} {m(v)}" for k, v in a.portfolio_contributions["total"].sort_values(ascending=False).items()))
        if fw == "CECL":
            lc = a.loan_contributions
            top = lc.groupby("counterparty_id")["ecl_change"].sum().sort_values(ascending=False).head(5)
            print("  Top borrowers:  ", ", ".join(f"{k} {m(v)}" for k, v in top.items()))

    p = config.path("outputs") / "phase1_attribution.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nWrote {p}")


if __name__ == "__main__":
    main()
