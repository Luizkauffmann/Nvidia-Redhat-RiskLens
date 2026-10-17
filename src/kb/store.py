"""Read-only access to the local knowledge base used by the regulatory and scenario-factor tools.

Runs entirely on the machine: PostgreSQL (pgvector + full-text search) and Ollama embeddings
(`nomic-embed-text`). Build it with `python -m scripts.load_kb`. Settings come from env:

    RISKLENS_KB_DSN      libpq connection string        (default "dbname=risklens")
    RISKLENS_EMBED_URL   Ollama base URL                (default http://localhost:11434)
    RISKLENS_EMBED_MODEL embedding model                (default nomic-embed-text)

psycopg is imported lazily, so the rest of the repo works on machines without Postgres; the tools
then return a KBUnavailable error with a fix hint instead of crashing.
"""
from __future__ import annotations

import hashlib
import json
import os
import urllib.error
import urllib.request
from typing import Any


class KBUnavailable(RuntimeError):
    """The knowledge base (Postgres or the embedding model) cannot be reached."""


def dsn() -> str:
    return os.environ.get("RISKLENS_KB_DSN", "dbname=risklens")


def connect():
    try:
        import psycopg
        from pgvector.psycopg import register_vector
    except ImportError as exc:
        raise KBUnavailable("psycopg / pgvector are not installed (pip install -r requirements.txt)") from exc
    try:
        conn = psycopg.connect(dsn(), connect_timeout=3)
    except psycopg.OperationalError as exc:
        raise KBUnavailable(f"cannot connect to the knowledge base ({dsn()}): start Postgres and run "
                            "`python -m scripts.load_kb`") from exc
    register_vector(conn)
    return conn


def embed(texts: list[str], kind: str) -> list[list[float]]:
    """nomic-embed-text expects a task prefix: 'search_document' or 'search_query'."""
    url = os.environ.get("RISKLENS_EMBED_URL", "http://localhost:11434").rstrip("/") + "/api/embed"
    model = os.environ.get("RISKLENS_EMBED_MODEL", "nomic-embed-text")
    body = json.dumps({"model": model, "input": [f"{kind}: {t}" for t in texts]}).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read())["embeddings"]
    except (urllib.error.URLError, KeyError, TimeoutError) as exc:
        raise KBUnavailable(f"embedding model unavailable at {url}: start Ollama and `ollama pull {model}`") from exc


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _rows(conn, sql: str, params: Any = None) -> list[dict[str, Any]]:
    from psycopg.rows import dict_row
    with conn.cursor(row_factory=dict_row) as cur:
        return cur.execute(sql, params).fetchall()


# ---------------------------------------------------------------- regulatory guidance

PASSAGE_COLS = """c.paragraph_ref as citation, d.title as document, d.issuer, d.framework, d.authority,
  d.effective_from::text as effective_from, d.source_file, d.url, c.heading, c.topics, c.cross_refs, c.body"""
FILTERS = """not c.superseded
  and (%(framework)s::text is null or d.framework = %(framework)s)
  and (%(topic)s::text is null or %(topic)s = any(c.topics))"""
HYBRID_SQL = f"""
with vec as (
  select c.id, row_number() over (order by c.embedding <=> %(qvec)s::vector) as rank
  from standard_chunks c join kb_documents d using (doc_id)
  where c.embedding is not null and {FILTERS}
  order by c.embedding <=> %(qvec)s::vector limit %(pool)s
), kw as (
  select c.id, row_number() over (order by ts_rank_cd(c.tsv, q) desc) as rank
  from standard_chunks c join kb_documents d using (doc_id), websearch_to_tsquery('english', %(qtext)s) q
  where c.tsv @@ q and {FILTERS}
  order by ts_rank_cd(c.tsv, q) desc limit %(pool)s
), fused as (
  -- Reciprocal rank fusion: rewards passages that rank well by meaning or by keywords, best by both.
  select id, sum(1.0 / (60 + rank)) as score from (select * from vec union all select * from kw) r group by id
)
select {PASSAGE_COLS}, round(f.score::numeric, 5)::float as score
from fused f join standard_chunks c using (id) join kb_documents d using (doc_id)
order by f.score desc limit %(k)s
"""


def search(query: str, k: int = 5, framework: str | None = None, topic: str | None = None) -> list[dict[str, Any]]:
    qvec = embed([query], "search_query")[0]
    with connect() as conn:
        return _rows(conn, HYBRID_SQL, {"qvec": qvec, "qtext": query, "framework": framework, "topic": topic,
                                        "pool": max(20, 4 * k), "k": k})


def passage(citation: str) -> list[dict[str, Any]]:
    with connect() as conn:
        return _rows(conn, f"select {PASSAGE_COLS}, c.superseded from standard_chunks c join kb_documents d using (doc_id) "
                           "where lower(c.paragraph_ref) = lower(%s)", (citation.strip(),))


def citing(reference: str) -> list[dict[str, Any]]:
    """Guidance passages that cite a codification reference such as '326-20-30-9' or '326'."""
    with connect() as conn:
        return _rows(conn, f"select {PASSAGE_COLS} from standard_chunks c join kb_documents d using (doc_id) "
                           "where %s = any(c.cross_refs) and not c.superseded order by d.doc_id, c.ordinal",
                     (reference.strip(),))


def documents() -> list[dict[str, Any]]:
    with connect() as conn:
        return _rows(conn, """select d.doc_id, d.title, d.issuer, d.framework, d.authority, d.effective_from::text as effective_from,
                                     d.licence, d.redistributable, d.url, d.source_file,
                                     count(c.id) as passages, bool_and(coalesce(c.superseded, false)) as superseded
                              from kb_documents d left join standard_chunks c using (doc_id)
                              group by d.doc_id order by d.framework, d.effective_from desc nulls last""")


# ---------------------------------------------------------------- scenario factors

def factors(scenario: str | None = None) -> list[dict[str, Any]]:
    with connect() as conn:
        rows = _rows(conn, "select scenario, segment as portfolio, pd_mult::float, lgd_adj::float, ead_ccf::float, source, notes "
                           "from scenario_factors where (%s::text is null or scenario = %s) order by scenario, segment",
                     (scenario, scenario))
    return rows
