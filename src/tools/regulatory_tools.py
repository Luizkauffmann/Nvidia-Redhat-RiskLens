"""Regulatory Agent tools: what the supervisory guidance says, with document + page citations.

Backed by the local knowledge base (src/kb, built by `python -m scripts.load_kb` from
docs/regulatory/sources). The tools return passages, never figures computed from them: they
answer "what does the rule say?", while the allowance numbers stay with the analytics tools.
"""
from __future__ import annotations

from typing import Any, Literal

from src.kb import store
from src.tools.base import ToolError, tool

REG_AGENTS = ["orchestrator", "regulatory_agent"]
Area = Literal["CECL", "IFRS9", "capital", "model_risk"]
TEXT_CHARS = 900  # keep results small for small models; get_regulatory_passage returns the full text


def _kb(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except store.KBUnavailable as exc:
        raise ToolError(str(exc), "dependency_error",
                        {"fix": "start Postgres and Ollama, then run `python -m scripts.load_kb`"}) from exc


def _passage(row: dict[str, Any], full: bool = False) -> dict[str, Any]:
    text = row["body"] if full or len(row["body"]) <= TEXT_CHARS else row["body"][:TEXT_CHARS].rsplit(" ", 1)[0] + " …"
    return {"citation": row["citation"], "document": row["document"], "issuer": row["issuer"], "area": row["framework"],
            "effective_from": row["effective_from"], "file": row["source_file"], "text": text,
            "topics": row["topics"], "codification_refs": row["cross_refs"]}


@tool("regulatory", REG_AGENTS)
def search_regulation(query: str, area: Area | None = None, k: int = 4) -> dict[str, Any]:
    """Search supervisory guidance (interagency ACL policy statement, CECL capital rules, SR 26-2 model risk, ECB/ESRB IFRS 9 papers) and return the best passages with citations like 'IPS 2023 p.12'; optional area filter."""
    k = max(1, min(int(k), 8))
    rows = _kb(store.search, query, k=k, framework=area)
    if not rows:
        raise ToolError(f"No guidance passages match {query!r}", hint={"try": "rephrase, drop the area filter, or call list_regulatory_documents"})
    return {"query": query, "area": area, "passages": [_passage(r) for r in rows],
            "note": "Cite passages as [citation]. Superseded guidance is excluded."}


@tool("regulatory", REG_AGENTS)
def get_regulatory_passage(citation: str) -> dict[str, Any]:
    """Full text of one guidance passage by its citation, e.g. 'IPS 2023 p.12' or 'SR 26-2 p.2 ¶2'."""
    rows = _kb(store.passage, citation)
    if not rows:
        raise ToolError(f"No passage with citation {citation!r}", hint={"fix": "use a citation returned by search_regulation"})
    r = rows[0]
    return {**_passage(r, full=True), "superseded": r["superseded"]}


@tool("regulatory", REG_AGENTS)
def find_guidance_citing(codification_ref: str) -> dict[str, Any]:
    """Guidance passages that cite a FASB codification reference, e.g. '326-20-30-9' (paragraph), '326-20' (subtopic) or '326' (topic)."""
    rows = _kb(store.citing, codification_ref)
    return {"codification_ref": codification_ref, "count": len(rows), "passages": [_passage(r) for r in rows[:10]],
            "note": "FASB ASC text itself is copyrighted and not in the knowledge base; these are the supervisory passages that cite it."}


@tool("regulatory", REG_AGENTS)
def list_regulatory_documents() -> dict[str, Any]:
    """The guidance documents in the knowledge base: issuer, area, effective date, licence, passage count and whether superseded."""
    docs = _kb(store.documents)
    return {"documents": [{**d, "passages": int(d["passages"])} for d in docs]}
