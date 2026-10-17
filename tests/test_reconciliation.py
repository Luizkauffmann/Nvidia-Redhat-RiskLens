from risklens import config
from risklens.tools import call_tool


def test_reconciliation_passes():
    r = call_tool("reconcile_allowance_change", {}).result
    assert r["status"] == "PASS"
    assert abs(r["residual"]) < 0.02
    assert r["opening_ties_to_booked"] and r["closing_ties_to_booked"]
    assert r["robustness"]["primary_driver_stable"]
    assert abs(sum(r["components"].values()) - r["explained_change"]) < 0.05


def test_thresholds_come_from_config(monkeypatch):
    cfg = dict(config.load("reconciliation"))
    cfg["pass_max_pct"] = -1.0          # nothing can pass
    cfg["warning_max_pct"] = -0.5       # nothing can warn either
    monkeypatch.setattr(config, "load", lambda name, _orig=config.load: cfg if name == "reconciliation" else _orig(name))
    assert call_tool("reconcile_allowance_change", {}).result["status"] == "FAIL"
