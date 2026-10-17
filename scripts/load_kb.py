"""Build the local knowledge base: regulatory guidance (page-cited passages) and scenario factors.

    createdb risklens                      # once; needs PostgreSQL with the pgvector extension
    ollama pull nomic-embed-text           # embedding model (274 MB)
    python -m scripts.load_kb              # schema + factors + every PDF in config/regulatory_sources.yaml
    python -m scripts.load_kb --dry-run    # parse and report, write nothing

Passages never span pages, so every citation is "<label> p.<page>" (docs/regulatory/README.md asks for
file + page). Only public-domain or redistribution-permitted documents are listed in the catalog.
"""
from __future__ import annotations

import argparse
import csv
import logging
import re
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.kb import store  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ROOT / "docs" / "regulatory" / "sources"
CATALOG = ROOT / "config" / "regulatory_sources.yaml"
FACTORS = ROOT / "data" / "kb" / "scenario_factors.csv"
SCHEMA = ROOT / "src" / "kb" / "schema.sql"
TARGET_CHARS = 1200   # ~300 tokens per passage

TOPICS = {
    "scope": [r"\bscope\b"],
    "collective_evaluation": [r"\bpool", r"collective", r"similar risk characteristics"],
    "historical_loss": [r"historical (credit )?loss", r"loss history"],
    "reasonable_supportable_forecast": [r"reasonable and supportable"],
    "reversion": [r"\brevert", r"reversion"],
    "qualitative_factors": [r"qualitative factor", r"\bq-factor"],
    "contractual_term": [r"contractual term", r"prepayment"],
    "collateral_dependent": [r"collateral[- ]dependent"],
    "pcd": [r"purchased (financial assets with )?credit[- ]deteriorat", r"\bpcd\b"],
    "off_balance_sheet": [r"off-balance-sheet", r"unfunded (loan )?commitment"],
    "overlays": [r"\boverlays?\b", r"post-model adjustment"],
    "sicr": [r"\bsicr\b", r"significant increase in credit risk", r"\bstage 2\b"],
    "novel_risks": [r"climate", r"geopolitic", r"novel risk"],
    "capital_transition": [r"transition(al)? (provision|amount|period)", r"phase[- ]in", r"\bcet1\b"],
    "model_risk": [r"model risk", r"model validation", r"effective challenge"],
    "write_off": [r"write[- ]?offs?\b", r"charge[- ]?offs?\b"],
    "disclosure": [r"disclos"],
    "documentation": [r"documentation"],
    "validation": [r"validat", r"back-?test"],
    "governance": [r"board of directors", r"internal control", r"senior management"],
    "examiner_review": [r"examiner"],
    "estimation_methods": [r"discounted cash flow", r"loss-rate", r"roll-rate", r"probability of default", r"vintage"],
}
TOPIC_RES = {t: [re.compile(p, re.I) for p in ps] for t, ps in TOPICS.items()}
XREF_RE = re.compile(r"\b(\d{3}-\d{2}-\d{2}-\d+[A-Z]?)\b")
XREF_SUB_RE = re.compile(r"\b(?:ASC\s+)?(?:Sub)?[Tt]opic\s+(\d{3}(?:-\d{2})?)\b|\bASC\s+(\d{3}-\d{2})\b(?!-\d)")


def tag_topics(text: str) -> list[str]:
    return [t for t, res in TOPIC_RES.items() if any(r.search(text) for r in res)]


def cross_refs(text: str) -> list[str]:
    return sorted(set(XREF_RE.findall(text)) | {a or b for a, b in XREF_SUB_RE.findall(text)})


def page_texts(path: Path) -> list[str]:
    from pypdf import PdfReader
    logging.getLogger("pypdf").setLevel(logging.ERROR)   # font-encoding warnings are noise here
    pages = [p.extract_text() or "" for p in PdfReader(path).pages]
    counts: dict[str, int] = {}
    for p in pages:   # running headers/footers: short lines repeated on more than a third of the pages
        for line in {ln.strip() for ln in p.splitlines() if 0 < len(ln.strip()) < 90}:
            key = re.sub(r"\d+", "#", line)
            counts[key] = counts.get(key, 0) + 1
    noise = {k for k, n in counts.items() if len(pages) > 3 and n > len(pages) / 3}
    out = []
    for p in pages:
        text = "\n".join(ln for ln in p.splitlines() if re.sub(r"\d+", "#", ln.strip()) not in noise)
        text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)      # words hyphenated across lines
        out.append(re.sub(r"\s+", " ", text).strip())
    return out


def split_page(text: str) -> list[str]:
    chunks, cur = [], ""
    for s in re.split(r"(?<=[.;:])\s+(?=[A-Z(\[§\d])", text):
        if cur and len(cur) + len(s) > TARGET_CHARS:
            chunks.append(cur)
            cur = s
        else:
            cur = f"{cur} {s}" if cur else s
    if cur:
        chunks.append(cur)
    return [c for c in chunks if len(c) > 80]


def parse_documents() -> list[tuple[dict, list[dict]]]:
    cat = yaml.safe_load(CATALOG.read_text())
    docs = []
    for fname, m in cat["documents"].items():
        path = SOURCES / fname
        if not path.exists():
            print(f"  missing {fname}: run scripts/fetch_regulatory_docs.sh")
            continue
        doc = {"doc_id": m["doc_id"], "title": m["title"], "issuer": m["issuer"], "standard": "GUIDANCE",
               "authority": "supervisory", "framework": m["framework"], "edition": fname,
               "effective_from": m.get("effective_from"), "url": m["url"], "licence": cat["licences"][m["licence"]],
               "redistributable": True, "source_file": f"docs/regulatory/sources/{fname}",
               "content_hash": store.sha(path.read_bytes().hex()[:200000])}
        chunks, ordinal = [], 0
        for page_no, text in enumerate(page_texts(path), start=1):
            for k, body in enumerate(split_page(text), start=1):
                chunks.append({"paragraph_ref": f"{m['label']} p.{page_no}" + (f" ¶{k}" if k > 1 else ""),
                               "heading": f"{m['label']}, page {page_no}", "section_path": f"{m['title']} > page {page_no}",
                               "ordinal": ordinal, "body": body, "topics": tag_topics(body), "cross_refs": cross_refs(body)})
                ordinal += 1
        docs.append((doc, chunks))
    return docs


DOC_COLS = ["doc_id", "title", "issuer", "standard", "authority", "framework", "edition", "effective_from", "url",
            "licence", "redistributable", "source_file", "content_hash"]
CHUNK_COLS = ["doc_id", "standard", "paragraph_ref", "heading", "section_path", "doc_part", "ordinal", "body", "topics",
              "cross_refs", "effective_from", "redistributable", "token_count", "content_hash", "embedding"]


def load_document(conn, doc: dict, chunks: list[dict], batch: int = 16) -> None:
    conn.execute(f"insert into kb_documents ({', '.join(DOC_COLS)}) values ({', '.join(['%s'] * len(DOC_COLS))}) "
                 f"on conflict (doc_id) do update set {', '.join(f'{c} = excluded.{c}' for c in DOC_COLS[1:])}, loaded_at = now()",
                 [doc.get(c) for c in DOC_COLS])
    conn.execute("delete from standard_chunks where doc_id = %s", (doc["doc_id"],))
    for i in range(0, len(chunks), batch):
        part = chunks[i:i + batch]
        vectors = store.embed([f"{c['heading']}\n{c['body']}" for c in part], "search_document")
        rows = [[doc["doc_id"], doc["standard"], c["paragraph_ref"], c["heading"], c["section_path"], "supervisory_guidance",
                 c["ordinal"], c["body"], c["topics"], c["cross_refs"], doc.get("effective_from"), True,
                 len(c["body"]) // 4, store.sha(c["body"]), v] for c, v in zip(part, vectors)]
        with conn.cursor() as cur:
            cur.executemany(f"insert into standard_chunks ({', '.join(CHUNK_COLS)}) "
                            f"values ({', '.join(['%s'] * len(CHUNK_COLS))})", rows)
    conn.commit()


def load_factors(conn) -> int:
    with open(FACTORS, newline="") as fh:
        rows = list(csv.DictReader(fh))
    conn.execute("delete from scenario_factors")
    with conn.cursor() as cur:
        cur.executemany("insert into scenario_factors (scenario, segment, pd_mult, lgd_adj, ead_ccf, source, notes) "
                        "values (%s, %s, %s, %s, %s, %s, %s)",
                        [(r["scenario"], r["portfolio"], float(r["pd_mult"]), float(r["lgd_adj"]),
                          float(r["ead_ccf"]) if r["ead_ccf"] else None, r["source"], r["notes"]) for r in rows])
    conn.commit()
    return len(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dry-run", action="store_true", help="parse and report; write nothing")
    a = ap.parse_args()
    docs = parse_documents()
    for doc, chunks in docs:
        print(f"{doc['doc_id']:<20} {len(chunks):>4} passages  {doc['framework']:<10} {doc['title']}")
    if a.dry_run:
        print("dry run: nothing written")
        return
    with store.connect() as conn:
        conn.execute(SCHEMA.read_text())
        print(f"scenario factors: {load_factors(conn)} rows from {FACTORS.relative_to(ROOT)}")
        for doc, chunks in docs:
            load_document(conn, doc, chunks)
            print(f"loaded {doc['doc_id']}")
    print(f"knowledge base ready: {sum(len(c) for _, c in docs)} passages from {len(docs)} documents")


if __name__ == "__main__":
    main()
