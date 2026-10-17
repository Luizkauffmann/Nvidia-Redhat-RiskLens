"""The traditional (pre-agent) CECL production workflow, as reusable steps.

    1. seed_source_data   load / create the bank's source extracts
                          (loan & borrower snapshots, macro scenarios, weights,
                          overlays, model registry)
    2. validate_inputs    data-quality and governance checks before a run
    3. run_official       score PD / LGD / EAD with the assigned model versions,
                          weight the scenarios, add overlays, and book the
                          official run (results + summary + run log)

This is the part that, in a bank, lives in a platform such as Dataiku or a
vendor engine. RiskLens agents never run it; they investigate its outputs
through src/tools. The Colab notebook drives these steps against a Google
Drive folder (see notebooks/RiskLens_CECL_Workflow.ipynb).
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.cecl.engine import ECLEngine
from src.data.loader import DataStore, write_ground_truth
from src.utils import config

SOURCE_TABLES = ["loans", "counterparties", "macro", "weights", "overlays",
                 "model_metadata", "model_assignments", "model_parameters"]
PRODUCTION_STATUSES = {"ACTIVE"}
REQUIRED_LOAN_FIELDS = ["loan_id", "counterparty_id", "portfolio", "product_type", "current_balance",
                        "pd_model_id", "lgd_model_id", "ead_model_id"]


# --------------------------------------------------------------------------- 1. source data
def seed_source_data(data_dir: Path | str | None = None, ground_truth_dir: Path | str | None = None,
                     overwrite: bool = False) -> dict[str, Any]:
    """Write the synthetic source extracts (stand-in for the bank's systems) and the evaluation labels.

    Skips if the loan snapshot already exists unless ``overwrite``. Seeded, so
    re-running produces identical files.
    """
    from src.data.generator import DemoDataGenerator, to_jsonable

    store = DataStore(data_dir)
    if (store.dir / "loan_snapshots.parquet").exists() and not overwrite:
        return {"status": "skipped", "reason": "source data already present", "data_dir": str(store.dir)}
    data = DemoDataGenerator().generate()
    for name in SOURCE_TABLES:
        attr = {"macro": "macro", "weights": "scenario_weights"}.get(name, name)
        store.write(name, getattr(data, attr))
    gt_path = write_ground_truth(to_jsonable(data.ground_truth), Path(ground_truth_dir) if ground_truth_dir else None)
    return {"status": "written", "data_dir": str(store.dir), "ground_truth": str(gt_path),
            "tables": inventory(store)}


def inventory(store: DataStore) -> list[dict[str, Any]]:
    """Rows, columns and dates per table in the data folder."""
    from src.data.loader import FILES
    out = []
    for name, fname in FILES.items():
        p = store.dir / fname
        if not p.exists():
            out.append({"table": name, "file": fname, "present": False})
            continue
        df = getattr(store, name)
        dates = sorted(f"{d:%Y-%m-%d}" for d in pd.to_datetime(df["as_of_date"]).unique()) if "as_of_date" in df else []
        out.append({"table": name, "file": fname, "present": True, "rows": len(df), "columns": df.shape[1],
                    "as_of_dates": dates, "size_kb": round(p.stat().st_size / 1024, 1)})
    return out


def export_model_parameters(models_cfg: dict[str, Any] | None = None) -> pd.DataFrame:
    """Flatten config/models.yaml into the model_parameters table (one row per parameter)."""
    models_cfg = models_cfg or config.models()
    rows = []
    for kind in ("pd_models", "lgd_models"):
        for mid, spec in models_cfg[kind].items():
            for k in ("intercept", "base", "floor", "cap"):
                if k in spec:
                    rows.append({"model_id": mid, "parameter": k, "value": spec[k]})
            for f, fs in spec.get("features", {}).items():
                rows.append({"model_id": mid, "parameter": f"{f}.coef", "value": fs["coef"]})
                rows.append({"model_id": mid, "parameter": f"{f}.center", "value": fs["center"]})
            for v, b in spec.get("macro_sensitivity", {}).items():
                rows.append({"model_id": mid, "parameter": f"macro.{v}", "value": b})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- 2. validation
def validate_inputs(as_of_date: str, store: DataStore | None = None) -> dict[str, Any]:
    """Pre-run controls. FAIL blocks the official run; WARNING is reported."""
    store = store or DataStore()
    checks: list[dict[str, Any]] = []

    def add(name: str, ok: bool, detail: Any = None, severity: str = "FAIL"):
        checks.append({"check": name, "status": "PASS" if ok else severity, "detail": detail})

    loans = store.loans_at(as_of_date)
    add("loans_present", len(loans) > 0, {"loans": len(loans)})
    missing = {c: int(loans[c].isna().sum()) for c in REQUIRED_LOAN_FIELDS if c in loans and loans[c].isna().any()}
    absent = [c for c in REQUIRED_LOAN_FIELDS if c not in loans]
    add("required_fields_populated", not missing and not absent, {"null_counts": missing, "missing_columns": absent})
    revolving = {p for p, v in config.methodology()["products"].items() if v["amortization"] == "REVOLVING"}
    no_term = loans["remaining_term_months"].isna() & ~loans["product_type"].isin(revolving)
    add("term_present_for_non_revolving", not no_term.any(),
        {"missing": int(no_term.sum()), "note": "revolving products use a behavioural life"})
    add("unique_loan_ids", loans["loan_id"].is_unique, {"duplicates": int(loans["loan_id"].duplicated().sum())})
    add("non_negative_balances", bool((loans["current_balance"] >= 0).all()),
        {"negative": int((loans["current_balance"] < 0).sum())})
    cps = store.counterparties_at(as_of_date)
    orphans = set(loans["counterparty_id"]) - set(cps["counterparty_id"])
    add("every_loan_has_a_borrower", not orphans, {"orphan_counterparties": len(orphans)}, "WARNING")

    method = config.methodology()
    unknown_products = sorted(set(loans["product_type"]) - set(method["products"]))
    add("products_in_methodology", not unknown_products, {"unknown": unknown_products})

    meta = store.model_metadata.set_index("model_id")
    models_cfg = config.models()
    used = sorted(set(loans["pd_model_id"]) | set(loans["lgd_model_id"]) | set(loans["ead_model_id"]))
    not_registered = [m for m in used if m not in meta.index]
    not_prod = {m: meta.loc[m, "status"] for m in used if m in meta.index and meta.loc[m, "status"] not in PRODUCTION_STATUSES}
    no_params = [m for m in set(loans["pd_model_id"]) | set(loans["lgd_model_id"])
                 if m not in models_cfg["pd_models"] and m not in models_cfg["lgd_models"]]
    add("models_registered", not not_registered, {"unregistered": not_registered})
    add("models_approved_for_production", not not_prod, {"not_active": not_prod})
    add("model_parameters_available", not no_params, {"missing": sorted(no_params)})
    asg = store.model_assignments_at(as_of_date).set_index("product_type")
    mism = 0
    for col in ("pd_model_id", "lgd_model_id", "ead_model_id"):
        mism += int((loans[col].to_numpy() != loans["product_type"].map(asg[col]).to_numpy()).sum())
    add("loan_models_match_assignments", mism == 0, {"mismatched_loans": mism})

    w = store.weights_at(as_of_date)
    scen = set(method["macro"]["scenarios"])
    add("scenario_weights_sum_to_one", abs(sum(w.values()) - 1.0) < 1e-9, {"weights": w})
    add("all_scenarios_weighted", set(w) == scen, {"weighted": sorted(w), "expected": sorted(scen)})
    wt = store.weights[pd.to_datetime(store.weights["as_of_date"]) == pd.Timestamp(as_of_date)]
    add("scenario_weights_approved", bool(wt["approved_by"].notna().all()),
        {"approved_by": wt["approved_by"].iloc[0] if len(wt) else None})
    macro = store.macro_at(as_of_date)
    rs = method["macro"]["reasonable_supportable_quarters"]
    q = macro.groupby("scenario")["forecast_quarter"].nunique().to_dict()
    add("macro_paths_cover_rs_period", all(q.get(s, 0) >= rs for s in scen), {"quarters": q, "required": rs})
    add("macro_variables_present", set(method["macro"]["variables"]) <= set(macro.columns),
        {"missing": sorted(set(method["macro"]["variables"]) - set(macro.columns))})
    ov = store.overlays_at(as_of_date)
    add("overlays_documented", bool(ov["reason"].notna().all()) if len(ov) else True,
        {"overlays": len(ov), "total": float(ov["amount"].sum()) if len(ov) else 0.0}, "WARNING")

    rank = {"PASS": 0, "WARNING": 1, "FAIL": 2}
    overall = max((c["status"] for c in checks), key=rank.get)
    return {"as_of_date": as_of_date, "status": overall, "checks": checks}


# --------------------------------------------------------------------------- 3. official run
def _frame_hash(df: pd.DataFrame) -> str:
    return hashlib.sha256(pd.util.hash_pandas_object(df.reset_index(drop=True), index=False).values.tobytes()).hexdigest()[:16]


def _upsert(store: DataStore, name: str, new: pd.DataFrame, as_of: pd.Timestamp, framework: str) -> pd.DataFrame:
    try:
        old = getattr(store, name)
        keep = ~((pd.to_datetime(old["as_of_date"]) == as_of) & (old["framework"] == framework))
        out = pd.concat([old[keep], new], ignore_index=True)
    except FileNotFoundError:
        out = new
    fw_order = {f: i for i, f in enumerate(config.settings()["frameworks"])}
    out = (out.assign(_d=pd.to_datetime(out["as_of_date"]), _f=out["framework"].map(fw_order))
           .sort_values(["_d", "_f"], kind="stable").drop(columns=["_d", "_f"]).reset_index(drop=True))
    store.write(name, out)
    return out


def run_official(as_of_date: str, frameworks: list[str] | None = None, data_dir: Path | str | None = None,
                 run_label: str = "OFFICIAL", allow_warnings: bool = True,
                 engine: ECLEngine | None = None) -> dict[str, Any]:
    """Validate inputs, calculate the allowance and book it as the official run for that date."""
    store = DataStore(data_dir)
    frameworks = frameworks or config.settings()["frameworks"]
    validation = validate_inputs(as_of_date, store)
    if validation["status"] == "FAIL" or (validation["status"] == "WARNING" and not allow_warnings):
        failed = [c for c in validation["checks"] if c["status"] != "PASS"]
        raise RuntimeError(f"Input validation {validation['status']} for {as_of_date}: {json.dumps(failed, default=str)}")
    engine = engine or ECLEngine()
    store.write("model_parameters", export_model_parameters())      # registry export = parameters actually used
    as_of = pd.Timestamp(as_of_date)
    inputs = store.engine_inputs(as_of_date)
    asg = store.model_assignments_at(as_of_date)
    booked = []
    for fw in frameworks:
        res = engine.run(inputs, fw)
        s = res.summary()
        s.update({f"scenario_ecl_{k}": v for k, v in s.pop("scenario_ecl").items()})
        s.update({f"weight_{k}": v for k, v in s.pop("scenario_weights").items()})
        run_id = f"{fw}_{as_of:%Y%m%d}_{run_label}"
        s["run_id"] = run_id
        s["scenario_set_id"] = store.scenario_set_id(as_of_date)
        loans = res.loans.copy()
        loans["as_of_date"] = as_of
        _upsert(store, "results", loans, as_of, fw)
        summ = pd.DataFrame([s])
        summ["as_of_date"] = as_of
        _upsert(store, "summary", summ, as_of, fw)
        log = {
            "run_id": run_id, "as_of_date": f"{as_of:%Y-%m-%d}", "framework": fw,
            "executed_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "allowance": float(res.allowance), "modeled_ecl": float(res.total_ecl), "overlay_total": float(res.overlay_total),
            "loan_count": int(len(res.loans)), "scenario_set_id": s["scenario_set_id"],
            "scenario_weights": json.dumps(res.weights),
            "model_versions": json.dumps({r.product_type: [r.pd_model_id, r.lgd_model_id, r.ead_model_id]
                                          for r in asg.itertuples()}),
            "input_hash_loans": _frame_hash(inputs.loans), "input_hash_macro": _frame_hash(inputs.macro),
            "validation_status": validation["status"],
        }
        _append_run_log(store, log)
        booked.append({k: log[k] for k in ("run_id", "framework", "allowance", "modeled_ecl", "overlay_total",
                                            "loan_count", "validation_status")})
    return {"as_of_date": as_of_date, "runs": booked, "validation": validation}


def _append_run_log(store: DataStore, row: dict[str, Any]) -> None:
    p = store.dir / "run_log.csv"
    df = pd.DataFrame([row])
    if p.exists():
        df = pd.concat([pd.read_csv(p), df], ignore_index=True)
    df.to_csv(p, index=False)


def run_log(store: DataStore | None = None) -> pd.DataFrame:
    p = (store or DataStore()).dir / "run_log.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def record_observed_ground_truth(previous_date: str, current_date: str, data_dir: Path | str | None = None,
                                 ground_truth_dir: Path | str | None = None) -> dict[str, Any]:
    """Add the observed CECL attribution to the evaluation labels (evaluation only)."""
    from src.cecl.attribution import AllowanceAttributor
    from src.data.generator import to_jsonable
    from src.data.loader import read_ground_truth

    gt_dir = Path(ground_truth_dir) if ground_truth_dir else None
    store = DataStore(data_dir)
    attr = AllowanceAttributor(store).run(previous_date, current_date, "CECL")
    by_cp = attr.loan_contributions.groupby("counterparty_id")["ecl_change"].sum().sort_values(ascending=False)
    gt = read_ground_truth(gt_dir)
    gt["observed_cecl"] = {
        "previous_allowance": attr.summary["previous_allowance"],
        "current_allowance": attr.summary["current_allowance"],
        "change": attr.summary["change"],
        "drivers": {d["driver"]: d["impact"] for d in attr.summary["drivers"]},
        "macro_variables": {m["variable"]: m["impact"] for m in attr.summary["macro_variables"]},
        "portfolio_change": attr.portfolio_contributions["total"].to_dict(),
        "top_borrowers_by_ecl_change": by_cp.head(10).to_dict(),
        "residual": attr.summary["residual"],
    }
    write_ground_truth(to_jsonable(gt), gt_dir)
    return gt


__all__ = ["seed_source_data", "inventory", "export_model_parameters", "validate_inputs", "run_official",
           "run_log", "record_observed_ground_truth"]


# --------------------------------------------------------------------------- 4. readable databank export
DICTIONARY: dict[str, dict[str, str]] = {
    "loan_portfolio": {
        "as_of_date": "Reporting (month-end) date of the snapshot",
        "loan_id": "Loan / facility identifier", "counterparty_id": "Borrower identifier (joins to borrowers)",
        "portfolio": "Reporting portfolio (CRE, C&I, Mortgage, Auto, Consumer, Credit Card)",
        "product_type": "Product as defined in methodology.yaml", "segment": "RETAIL or COMMERCIAL",
        "property_type": "CRE collateral type (Office, Retail, Multifamily, Industrial, ...)",
        "origination_date": "Booking date", "maturity_date": "Contractual maturity (blank for revolving cards)",
        "original_term_months": "Contractual term at origination", "remaining_term_months": "Months to maturity",
        "original_balance": "Funded amount at origination (USD)", "current_balance": "Drawn balance (USD)",
        "credit_limit": "Commitment / limit for revolving products (USD)", "undrawn_amount": "Limit minus balance (USD)",
        "utilization": "Balance / limit", "interest_rate": "Contract rate (decimal); IFRS 9 discount rate",
        "collateral_value": "Latest collateral value (USD)", "ltv": "Loan-to-value (balance / collateral)",
        "dscr": "Debt service coverage ratio (commercial)", "watchlist_flag": "1 = on credit watchlist",
        "days_past_due": "Days past due", "delinquency_bucket": "CURRENT, 1-29, 30-59, 60-89, 90+",
        "default_flag": "1 = defaulted (PD = 100%)",
        "pd_model_id": "PD model version used", "lgd_model_id": "LGD model version used", "ead_model_id": "EAD model version used",
        "fico": "Credit score (retail)", "dti": "Debt-to-income (retail)",
        "rating_notch": "Internal rating notch, 1 = AAA ... 17 = CCC (commercial)", "internal_rating": "Internal rating grade",
        "leverage": "Debt / EBITDA (C&I)", "industry": "Borrower industry", "state": "US state",
        "scenario_set_id": "Macro scenario set used for the run",
        "origination_fico": "FICO at origination", "origination_rating_notch": "Rating notch at origination",
        "origination_pd": "Base PD at origination (IFRS 9 SICR reference)",
    },
    "borrowers": {
        "as_of_date": "Reporting date", "counterparty_id": "Borrower identifier",
        "counterparty_type": "INDIVIDUAL or COMPANY", "customer_name": "Synthetic name",
        "industry": "Industry", "state": "US state", "annual_revenue": "Annual revenue (USD, companies)",
        "total_assets": "Total assets (USD, companies)", "employees": "Employees (companies)",
        "fico": "Credit score (individuals)", "income": "Annual income (USD, individuals)", "dti": "Debt-to-income",
        "rating_notch": "Internal rating notch (1 = AAA ... 17 = CCC)", "internal_rating": "Internal rating grade",
        "sp_rating": "S&P rating if rated", "moodys_rating": "Moody's rating if rated", "dscr": "Debt service coverage",
        "leverage": "Debt / EBITDA", "watchlist_flag": "1 = on watchlist", "economic_group_id": "Group of connected borrowers",
    },
    "ecl_results": {
        "as_of_date": "Reporting date", "framework": "CECL or IFRS9", "loan_id": "Loan identifier",
        "counterparty_id": "Borrower identifier", "portfolio": "Reporting portfolio", "product_type": "Product",
        "current_balance": "Drawn balance (USD)", "undrawn_amount": "Undrawn commitment (USD)",
        "ead": "Exposure at default (balance + CCF x undrawn, USD)",
        "pd_base_annual": "Scorecard PD without macro adjustment (borrower quality only)",
        "pd_12m": "Scenario-weighted 12-month PD including macro forecast",
        "pd_annual_y1": "Scenario-weighted annual PD, first projection year", "lifetime_pd": "Scenario-weighted lifetime PD",
        "lgd": "Scenario-weighted LGD, first projection year", "horizon_years": "Remaining life used (years)",
        "ifrs9_stage": "IFRS 9 stage 1/2/3", "measurement": "12M or LIFETIME", "defaulted": "True if in default",
        "ecl": "Probability-weighted expected credit loss (USD)",
        "ecl_baseline": "ECL under the baseline scenario (USD)", "ecl_adverse": "ECL under the adverse scenario (USD)",
        "ecl_severe_adverse": "ECL under the severe adverse scenario (USD)",
        "pd_model_id": "PD model version", "lgd_model_id": "LGD model version", "ead_model_id": "EAD model version",
        "scenario_set_id": "Macro scenario set",
    },
}


def export_databank(out_dir: Path | str, store: DataStore | None = None,
                    frameworks: list[str] | None = None) -> list[dict[str, Any]]:
    """Write the portfolio, borrowers and ECL results as CSV per reporting date, plus reference tables and a data dictionary.

    This is a readable copy for people (Drive preview, Google Sheets, Excel). The
    engine and the tools keep reading the Parquet files in the data folder.
    """
    store = store or DataStore()
    out = Path(out_dir)
    frameworks = frameworks or ["CECL"]
    written: list[dict[str, Any]] = []

    def save(df: pd.DataFrame, rel: str, what: str) -> None:
        p = out / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        df = df.copy()
        for c in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[c]):
                df[c] = df[c].dt.strftime("%Y-%m-%d")
        df.to_csv(p, index=False)
        written.append({"file": rel, "rows": len(df), "columns": df.shape[1], "size_mb": round(p.stat().st_size / 1e6, 2),
                        "contents": what})

    for d in store.available_dates():
        save(store.loans_at(d), f"{d}/loan_portfolio_{d}.csv", "Loan-level portfolio snapshot")
        save(store.counterparties_at(d), f"{d}/borrowers_{d}.csv", "Borrower (counterparty) snapshot")
        for fw in frameworks:
            try:
                save(store.results_at(d, fw), f"{d}/ecl_results_{fw}_{d}.csv", f"Official {fw} loan-level results")
            except (KeyError, FileNotFoundError):
                pass
    ref = {"reference/model_registry.csv": (store.model_metadata, "Model inventory: versions, status, dates"),
           "reference/model_assignments.csv": (store.model_assignments, "Which PD/LGD/EAD model each product used"),
           "reference/model_parameters.csv": (store.model_parameters, "Scorecard coefficients and LGD parameters"),
           "reference/macro_scenarios.csv": (store.macro, "Quarterly macro forecast paths by scenario"),
           "reference/scenario_weights.csv": (store.weights, "Scenario probabilities and approvals"),
           "reference/management_overlays.csv": (store.overlays, "Qualitative overlays"),
           "reference/allowance_summary.csv": (store.summary, "Official run totals by date and framework")}
    for rel, (df, what) in ref.items():
        save(df, rel, what)
    dict_rows = [{"table": t, "column": c, "description": desc} for t, cols in DICTIONARY.items() for c, desc in cols.items()]
    save(pd.DataFrame(dict_rows), "data_dictionary.csv", "Column definitions")
    save(pd.DataFrame(written), "manifest.csv", "List of files in this databank")
    return written


__all__.append("export_databank")
