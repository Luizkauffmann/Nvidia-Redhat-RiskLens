"""Credit Agent tools: PD / LGD / rating migration and borrower-level drill-down."""
from __future__ import annotations

import numpy as np
import pandas as pd

from risklens import config
from risklens.cecl import attribution as A
from risklens.cecl.engine import base_logit
from risklens.tools.common import check_order, loans, portfolios, store, wavg
from risklens.tools.registry import ToolError, tool


def _migration(col: str, prev: str, cur: str, portfolio: str | None, extra: str | None = None) -> dict:
    P, C = loans(prev, portfolio), loans(cur, portfolio)
    surv = P["loan_id"].isin(C["loan_id"])
    Ps = P[surv].set_index("loan_id")
    Cs = C.set_index("loan_id").loc[Ps.index]
    out_by = {}
    for p in (portfolios() if portfolio is None else [portfolio]):
        a, b = P[P["portfolio"] == p], C[C["portfolio"] == p]
        row = {"previous_pct": (wavg(a[col], a["ead"]) or 0) * 100, "current_pct": (wavg(b[col], b["ead"]) or 0) * 100}
        row["change_pp"] = row["current_pct"] - row["previous_pct"]
        out_by[p] = row
    res = {
        "measure": f"EAD-weighted {col}", "portfolio": portfolio or "ALL",
        "all_loans": {"previous_pct": (wavg(P[col], P["ead"]) or 0) * 100, "current_pct": (wavg(C[col], C["ead"]) or 0) * 100},
        "existing_loans_only": {"previous_pct": (wavg(Ps[col], Ps["ead"]) or 0) * 100, "current_pct": (wavg(Cs[col], Cs["ead"]) or 0) * 100},
        "by_portfolio": out_by,
    }
    for k in ("all_loans", "existing_loans_only"):
        res[k]["change_pp"] = res[k]["current_pct"] - res[k]["previous_pct"]
    res["largest_increase_portfolio"] = max(out_by, key=lambda p: out_by[p]["change_pp"])
    return res


@tool("credit")
def calculate_pd_migration(previous_date: str, current_date: str, portfolio: str = None) -> dict:
    """12-month PD in both periods, split into borrower-driven (scorecard) and macro-driven movement."""
    check_order(previous_date, current_date)
    res = _migration("pd_12m", previous_date, current_date, portfolio)
    # Scorecard-only PD (no macro) isolates borrower deterioration from forecast changes.
    s = store()
    for d, key in ((previous_date, "previous_pct"), (current_date, "current_pct")):
        df = loans(d, portfolio)
        run = s.run(d)
        base = pd.Series(run.pd_12m_base, index=run.loan_id).reindex(df["loan_id"]).to_numpy()
        nd = ~df["default_flag"].to_numpy()
        res.setdefault("scorecard_pd_no_macro", {})[key] = (wavg(base[nd], df["ead"].to_numpy()[nd]) or 0) * 100
    sc = res["scorecard_pd_no_macro"]
    sc["change_pp"] = sc["current_pct"] - sc["previous_pct"]
    res["interpretation_aid"] = "scorecard_pd_no_macro excludes the macro forecast and defaulted loans; its change reflects borrower attributes only"
    return res


@tool("credit")
def calculate_lgd_migration(previous_date: str, current_date: str, portfolio: str = None) -> dict:
    """LGD in both periods (EAD-weighted), total and by portfolio."""
    check_order(previous_date, current_date)
    return _migration("lgd", previous_date, current_date, portfolio)


@tool("credit")
def calculate_rating_migration(previous_date: str, current_date: str, portfolio: str = None) -> dict:
    """Internal-rating migration matrix for commercial loans held in both periods (counts and balance)."""
    check_order(previous_date, current_date)
    P = loans(previous_date, portfolio)
    C = loans(current_date, portfolio).set_index("loan_id")
    P = P[P["internal_rating"].notna() & P["loan_id"].isin(C.index)].set_index("loan_id")
    if P.empty:
        raise ToolError("No rated (commercial) loans in this selection.", {"rated_portfolios": ["CRE", "C&I"]})
    Cn = C.loc[P.index]
    d = (Cn["rating_notch"] - P["rating_notch"]).astype(int)
    mat = pd.crosstab(P["internal_rating"], Cn["internal_rating"])
    scale = config.methodology()["rating_scale"]
    mat = mat.reindex(index=[r for r in scale if r in mat.index], columns=[r for r in scale if r in mat.columns], fill_value=0)
    bal = Cn["current_balance"]
    return {"portfolio": portfolio or "ALL_RATED", "loans": int(len(P)),
            "downgraded": {"loans": int((d > 0).sum()), "balance": float(bal[d > 0].sum())},
            "upgraded": {"loans": int((d < 0).sum()), "balance": float(bal[d < 0].sum())},
            "unchanged": {"loans": int((d == 0).sum())},
            "downgrades_by_notches": {int(k): int(v) for k, v in d[d > 0].value_counts().sort_index().items()},
            "downgraded_balance_by_portfolio": Cn.loc[d > 0].groupby("portfolio")["current_balance"].sum().to_dict(),
            "matrix_previous_rows_current_columns": {r: {c: int(v) for c, v in row.items() if v} for r, row in mat.iterrows()}}


def _reasons(p: pd.DataFrame | None, c: pd.DataFrame | None) -> list[str]:
    if p is None or p.empty:
        return ["new loan(s) originated in the current period"]
    if c is None or c.empty:
        return ["loan(s) exited (paid off, matured or charged off)"]
    r = []
    p0, c0 = p.iloc[0], c.iloc[0]
    if pd.notna(p0.get("rating_notch")) and c0["rating_notch"] > p0["rating_notch"]:
        r.append(f"rating downgrade {p0['internal_rating']} -> {c0['internal_rating']} ({int(c0['rating_notch'] - p0['rating_notch'])} notches)")
    if pd.notna(p0.get("dscr")) and c0["dscr"] < p0["dscr"] - 0.03:
        r.append(f"DSCR {p0['dscr']:.2f} -> {c0['dscr']:.2f}")
    if pd.notna(p0.get("ltv")) and c0["ltv"] > p0["ltv"] + 0.02:
        r.append(f"LTV {p0['ltv']:.2f} -> {c0['ltv']:.2f}")
    if bool(c0["watchlist_flag"]) and not bool(p0["watchlist_flag"]):
        r.append("added to watchlist")
    if c0["days_past_due"] > p0["days_past_due"]:
        r.append(f"DPD {int(p0['days_past_due'])} -> {int(c0['days_past_due'])}")
    if pd.notna(p0.get("fico")) and c0["fico"] < p0["fico"] - 20:
        r.append(f"FICO {int(p0['fico'])} -> {int(c0['fico'])}")
    if not r:
        r.append("no borrower deterioration; change from macro forecast, weights or balance")
    return r


@tool("credit", enums={"level": ["counterparty", "loan"]})
def identify_top_contributors(previous_date: str, current_date: str, n: int = 10, portfolio: str = None,
                              level: str = "counterparty") -> dict:
    """Borrowers (or loans) ranked by allowance increase, with PD/LGD change, driver split and reason codes.

    n: number of rows to return (1-50)
    level: 'counterparty' (default) or 'loan'
    """
    check_order(previous_date, current_date)
    n = max(1, min(int(n), 50))
    steps = A.loan_level_steps(previous_date, current_date)
    P = store().loans(previous_date).set_index("loan_id")
    C = store().loans(current_date).set_index("loan_id")
    meta = pd.concat([C[["counterparty_id", "portfolio", "product_type", "property_type"]],
                      P.loc[P.index.difference(C.index), ["counterparty_id", "portfolio", "product_type", "property_type"]]])
    df = steps.join(meta)
    if portfolio:
        df = df[df["portfolio"] == portfolio]
    key = "counterparty_id" if level == "counterparty" else "loan_id"
    df = df.reset_index().rename(columns={"index": "loan_id"})
    agg_cols = ["runoff", "growth", "credit", "macro", "methodology", "ecl_prev", "ecl_cur"]
    g = df.groupby(key)[agg_cols].sum()
    g["ecl_change"] = g["ecl_cur"] - g["ecl_prev"]
    top = g.nlargest(n, "ecl_change")
    rows = []
    for k, r in top.iterrows():
        sel_p = P[P["counterparty_id"] == k] if level == "counterparty" else P.loc[[k]] if k in P.index else None
        sel_c = C[C["counterparty_id"] == k] if level == "counterparty" else C.loc[[k]] if k in C.index else None
        cp_id = k if level == "counterparty" else df.loc[df["loan_id"] == k, "counterparty_id"].iloc[0]
        name = store().counterparties.set_index("counterparty_id").loc[cp_id, "customer_name"]
        rows.append({
            "counterparty_id" if level == "counterparty" else "loan_id": k, "customer_name": name,
            "portfolio": df.loc[df[key] == k, "portfolio"].iloc[0],
            "balance_current": float(sel_c["current_balance"].sum()) if sel_c is not None else 0.0,
            "ecl_previous": r["ecl_prev"], "ecl_current": r["ecl_cur"], "ecl_change": r["ecl_change"],
            "pd_12m_previous_pct": (wavg(sel_p["pd_12m"], sel_p["ead"]) or 0) * 100 if sel_p is not None and len(sel_p) else None,
            "pd_12m_current_pct": (wavg(sel_c["pd_12m"], sel_c["ead"]) or 0) * 100 if sel_c is not None and len(sel_c) else None,
            "lgd_previous_pct": (wavg(sel_p["lgd"], sel_p["ead"]) or 0) * 100 if sel_p is not None and len(sel_p) else None,
            "lgd_current_pct": (wavg(sel_c["lgd"], sel_c["ead"]) or 0) * 100 if sel_c is not None and len(sel_c) else None,
            "driver_split": {d: float(r[d]) for d in ("runoff", "growth", "credit", "macro", "methodology")},
            "reasons": _reasons(sel_p, sel_c),
        })
    return {"level": level, "portfolio": portfolio or "ALL", "ranked_by": "allowance (ECL) increase", "rows": rows,
            "top_n_share_of_total_change_pct": float(top["ecl_change"].sum() / g["ecl_change"].sum() * 100) if g["ecl_change"].sum() else None}


@tool("credit")
def get_loan_details(loan_id: str, as_of_date: str = None) -> dict:
    """All fields of one loan; both periods unless as_of_date is given."""
    s = store()
    out = {}
    for d in ([as_of_date] if as_of_date else s.dates):
        df = s.loans(d)
        row = df[df["loan_id"] == loan_id]
        out[d] = row.iloc[0].to_dict() if len(row) else None
    if all(v is None for v in out.values()):
        raise ToolError(f"Loan '{loan_id}' not found.", "Loan ids look like LN_0000123; use identify_top_contributors(level='loan') to find them.")
    return {"loan_id": loan_id, "by_date": out}


@tool("credit")
def get_counterparty_details(counterparty_id: str) -> dict:
    """Borrower profile and its loans with balance, PD, LGD and ECL in each period."""
    s = store()
    cps = s.counterparties.set_index("counterparty_id")
    if counterparty_id not in cps.index:
        raise ToolError(f"Counterparty '{counterparty_id}' not found.", "Ids look like CP_000123; use identify_top_contributors to find them.")
    by_date = {}
    for d in s.dates:
        df = s.loans(d)
        sub = df[df["counterparty_id"] == counterparty_id]
        by_date[d] = {"loans": sub[["loan_id", "product_type", "property_type", "current_balance", "internal_rating", "dscr", "ltv",
                                    "watchlist_flag", "days_past_due", "pd_12m", "lgd", "ecl"]].to_dict("records"),
                      "total_balance": float(sub["current_balance"].sum()), "total_ecl": float(sub["ecl"].sum())}
    return {"counterparty_id": counterparty_id, "profile": cps.loc[counterparty_id].to_dict(), "by_date": by_date}


@tool("credit")
def compare_borrower_periods(counterparty_id: str, previous_date: str = None, current_date: str = None) -> dict:
    """Explain a borrower's PD change: attribute changes and the PD-logit contribution of each scorecard feature."""
    s = store()
    previous_date = previous_date or s.dates[0]
    current_date = current_date or s.dates[-1]
    P = s.loans(previous_date)
    C = s.loans(current_date)
    p = P[P["counterparty_id"] == counterparty_id].set_index("loan_id")
    c = C[C["counterparty_id"] == counterparty_id].set_index("loan_id")
    common = p.index.intersection(c.index)
    if len(common) == 0:
        raise ToolError(f"Counterparty '{counterparty_id}' has no loan held in both periods.",
                        "Use get_counterparty_details to see its loans.")
    lid = c.loc[common, "current_balance"].idxmax()
    a, b = p.loc[lid], c.loc[lid]
    params = config.models()["pd_models"][b["pd_model_id"]]
    feats = {}
    if params["type"] == "commercial":
        feats = {"rating_notch": params["rating"] * (b["rating_notch"] - a["rating_notch"]),
                 "dscr": params["dscr"] * (b["dscr"] - a["dscr"]),
                 "ltv": params["ltv"] * (np.nan_to_num(b["ltv"], nan=0.65) - np.nan_to_num(a["ltv"], nan=0.65)),
                 "leverage": params["leverage"] * (np.nan_to_num(b["leverage"], nan=3) - np.nan_to_num(a["leverage"], nan=3))}
    else:
        feats = {"fico": params["fico"] * (b["fico"] - a["fico"]) / 100, "dti": params["dti"] * (b["dti"] - a["dti"]),
                 "ltv": params["ltv"] * (np.nan_to_num(b["ltv"], nan=0.8) - np.nan_to_num(a["ltv"], nan=0.8)),
                 "utilization": params["util"] * (np.nan_to_num(b["utilization"], nan=0.3) - np.nan_to_num(a["utilization"], nan=0.3))}
    la = base_logit(p.loc[[lid]].reset_index())[0]
    lb = base_logit(c.loc[[lid]].reset_index())[0]
    feats["watchlist_and_delinquency"] = (lb - la) - sum(feats.values())
    sig = lambda x: 1 / (1 + np.exp(-x))  # noqa: E731
    return {"counterparty_id": counterparty_id, "loan_id": lid, "pd_model_id": b["pd_model_id"],
            "attributes": {f: {"previous": a[f], "current": b[f]} for f in
                           ("internal_rating", "rating_notch", "dscr", "ltv", "leverage", "fico", "dti", "utilization",
                            "watchlist_flag", "days_past_due") if pd.notna(a[f]) or pd.notna(b[f])},
            "scorecard_pd_previous_pct": sig(la) * 100, "scorecard_pd_current_pct": sig(lb) * 100,
            "pd_12m_with_macro_previous_pct": float(a["pd_12m"]) * 100, "pd_12m_with_macro_current_pct": float(b["pd_12m"]) * 100,
            "logit_change_by_feature": feats, "logit_change_total": lb - la,
            "note": "Logit contributions are exact and additive for the scorecard; the macro effect is the gap between scorecard PD and PD with macro."}
