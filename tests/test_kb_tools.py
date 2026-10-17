"""Regulatory and scenario-factor tools.

Most tests replace the knowledge-base store with in-memory fakes, so they run in CI without Postgres or Ollama.
Tests marked `needs_kb` run against the real local knowledge base and are skipped when it is not built.
"""
from __future__ import annotations

import csv
import os
from pathlib import Path

import pytest

from src.kb import store
from src.tools import ToolContext, call_tool, list_tools, set_context
from src.tools.factor_tools import FactorScenario
from src.utils import config
from tests.conftest import CURR, PREV

ROOT = Path(__file__).resolve().parents[1]
FACTORS_CSV = ROOT / "data" / "kb" / "scenario_factors.csv"


def _kb_ready() -> bool:
    try:
        return bool(store.documents()) and bool(store.factors())
    except store.KBUnavailable:
        return False


needs_kb = pytest.mark.skipif(not _kb_ready(), reason="local knowledge base not built (python -m scripts.load_kb)")


@pytest.fixture(scope="module", autouse=True)
def ctx(generated, attribution, tmp_path_factory):
    c = ToolContext(data_dir=generated[0], cache_dir=tmp_path_factory.mktemp("cache"), disk_cache=False)
    c._mem[f"attr_default_CECL_{PREV}_{CURR}"] = attribution
    set_context(c)
    old = os.environ.get("RISKLENS_OUTPUT_DIR")
    os.environ["RISKLENS_OUTPUT_DIR"] = str(tmp_path_factory.mktemp("outputs"))
    config.reload()
    yield c
    set_context(None)
    if old is None:
        os.environ.pop("RISKLENS_OUTPUT_DIR", None)
    else:
        os.environ["RISKLENS_OUTPUT_DIR"] = old
    config.reload()


def _csv_factors(scenario=None):
    with open(FACTORS_CSV, newline="") as fh:
        rows = [{"scenario": r["scenario"], "portfolio": r["portfolio"], "pd_mult": float(r["pd_mult"]),
                 "lgd_adj": float(r["lgd_adj"]), "ead_ccf": float(r["ead_ccf"]) if r["ead_ccf"] else None,
                 "source": r["source"], "notes": r["notes"]} for r in csv.DictReader(fh)]
    return [r for r in rows if scenario is None or r["scenario"] == scenario]


@pytest.fixture
def csv_store(monkeypatch):
    monkeypatch.setattr(store, "factors", _csv_factors)


# --------------------------------------------------------------------------- scenario factors
def test_factor_scenarios_match_the_seed_file():
    assert {r["scenario"] for r in _csv_factors()} == set(FactorScenario.__args__)
    assert all(r["source"].startswith("Illustrative") for r in _csv_factors())


def test_baseline_factors_change_nothing(csv_store, ctx):
    r = call_tool("apply_scenario_factors", {"factor_scenario": "baseline", "as_of_date": CURR}, log=None)
    assert r.status == "ok"
    assert r.result["change"] == pytest.approx(0.0, abs=1e-6)
    assert r.result["scenario_allowance"] == pytest.approx(r.result["official_allowance"])


def test_factor_what_if_is_deterministic_and_adds_up(csv_store, ctx):
    args = {"factor_scenario": "fed_2026_severely_adverse", "as_of_date": "Aug 2026"}   # date is normalised
    a = call_tool("apply_scenario_factors", args, log=None).result
    b = call_tool("apply_scenario_factors", args, log=None).result
    assert a["scenario_allowance"] == b["scenario_allowance"]
    assert a["scenario_allowance"] == pytest.approx(sum(p["scenario_allowance"] for p in a["by_portfolio"]))
    assert a["change"] > 0
    cre = next(p for p in a["by_portfolio"] if p["portfolio"] == "CRE")
    expected = cre["modeled_ecl"] * cre["pd_mult_applied"] * cre["lgd_scenario"] / cre["lgd"]
    assert cre["scenario_modeled_ecl"] == pytest.approx(expected)
    assert cre["scenario_allowance"] == pytest.approx(cre["scenario_modeled_ecl"] + cre["overlay"])  # overlay carried
    assert a["not_covered"] == ["Consumer"]                                                          # no invented factors
    consumer = next(p for p in a["by_portfolio"] if p["portfolio"] == "Consumer")
    assert consumer["change"] == 0.0 and consumer["covered"] is False


def test_stressed_pd_is_capped_at_100_percent(monkeypatch, ctx):
    huge = [{**r, "pd_mult": 1000.0} for r in _csv_factors("fed_2026_severely_adverse")]
    monkeypatch.setattr(store, "factors", lambda s=None: huge)
    x = call_tool("apply_scenario_factors", {"factor_scenario": "fed_2026_severely_adverse", "as_of_date": CURR}, log=None).result
    for p in x["by_portfolio"]:
        if p["covered"]:
            assert p["pd_mult_applied"] < 1000.0


def test_unknown_factor_scenario_gets_a_hint(ctx):
    r = call_tool("apply_scenario_factors", {"factor_scenario": "stagflation", "as_of_date": CURR}, log=None)
    assert r.status == "error" and "rates_plus_200bp" in str(r.error["hint"])


# --------------------------------------------------------------------------- regulatory tools
FAKE_PASSAGES = [{"citation": "IPS 2023 p.13", "document": "Interagency Policy Statement on Allowances for Credit Losses (revised April 2023)",
                  "issuer": "FRB, FDIC, NCUA, OCC", "framework": "CECL", "authority": "supervisory", "effective_from": "2023-04-01",
                  "source_file": "docs/regulatory/sources/us_interagency_policy_statement_ACL_rev2023.pdf", "url": "https://example",
                  "heading": "IPS 2023, page 13", "topics": ["reversion"], "cross_refs": ["326"],
                  "body": "FASB ASC Topic 326 does not require the application of a specific reversion technique. " * 30}]


def test_regulatory_tools_cite_and_trim(monkeypatch, ctx):
    monkeypatch.setattr(store, "search", lambda q, k=5, framework=None, topic=None: FAKE_PASSAGES[:k])
    r = call_tool("search_regulation", {"query": "reversion", "area": "CECL"}, log=None)
    assert r.status == "ok"
    p = r.result["passages"][0]
    assert p["citation"] == "IPS 2023 p.13" and p["file"].endswith(".pdf")
    assert len(p["text"]) < len(FAKE_PASSAGES[0]["body"])                       # trimmed for small models


def test_regulatory_tools_report_a_missing_kb(monkeypatch, ctx):
    def down(*a, **k):
        raise store.KBUnavailable("cannot connect to the knowledge base")
    monkeypatch.setattr(store, "search", down)
    r = call_tool("search_regulation", {"query": "reversion"}, log=None)
    assert r.status == "error" and r.error["error_type"] == "dependency_error" and "load_kb" in r.error["hint"]["fix"]


def test_regulatory_agent_owns_only_guidance_tools():
    assert list_tools("regulatory_agent") == ["search_regulation", "get_regulatory_passage", "find_guidance_citing",
                                              "list_regulatory_documents"]
    assert {"get_scenario_factors", "apply_scenario_factors"} <= set(list_tools("macro_agent"))


def test_mock_team_routes_guidance_and_what_if_questions(monkeypatch, csv_store, ctx):
    from src.agents.team import build_team
    from src.llm.mock_provider import MockProvider

    monkeypatch.setattr(store, "search", lambda q, k=5, framework=None, topic=None: FAKE_PASSAGES[:k])
    run = build_team(MockProvider()).run("What does the guidance say about reversion after the forecast period?")
    team = next(e for e in run.events if e["type"] == "team")
    assert "regulatory_agent" in team["specialists"] and "IPS 2023 p.13" in run.answer
    run = build_team(MockProvider()).run("What would the allowance be if unemployment hits 7%?")
    team = next(e for e in run.events if e["type"] == "team")
    assert "macro_agent" in team["specialists"] and "unemployment_7pct" in run.answer
    assert run.grounding["status"] == "PASS"


# --------------------------------------------------------------------------- loader + live knowledge base
def test_loader_cites_every_passage_by_page():
    from scripts.load_kb import parse_documents

    docs = parse_documents()
    assert {d["doc_id"] for d, _ in docs} >= {"ips_acl_2023", "sr26_2"}
    for doc, chunks in docs:
        assert doc["redistributable"] and chunks, doc["doc_id"]
        assert all(" p." in c["paragraph_ref"] for c in chunks)
        assert len({c["paragraph_ref"] for c in chunks}) == len(chunks)          # citations are unique


@needs_kb
def test_live_search_finds_the_current_reversion_guidance(ctx):
    r = call_tool("search_regulation", {"query": "what happens after the reasonable and supportable forecast period"}, log=None)
    assert r.status == "ok"
    assert any(p["citation"].startswith("IPS 2023 p.") for p in r.result["passages"])
    assert not any(p["citation"].startswith("IPS ") and "2023" not in p["citation"] for p in r.result["passages"])  # superseded excluded
