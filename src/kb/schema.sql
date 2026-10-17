-- Risk Lens local store: scenario factors (relational) + knowledge base for the audit agent.
-- Apply with: psql -d risklens -f sql/schema.sql   (safe to re-run; never drops data)

create extension if not exists vector;

-- Deterministic lookup for the ECL spreadsheet model. No embeddings: the same question must
-- always get the same numbers, and every number carries its source for the audit trail.
-- Same columns as the Factors sheet in demo/risklens_ecl_illustrative.xlsx.
create table if not exists scenario_factors (
  scenario   text    not null,           -- e.g. 'unemployment_7pct', 'fed_2026_severely_adverse'
  segment    text    not null,           -- 'cre', 'c_and_i', 'residential_mortgage', 'consumer_card', 'auto'
  pd_mult    numeric not null check (pd_mult > 0),
  lgd_adj    numeric not null default 0, -- additive adjustment to LGD, in fraction (0.05 = +5 points)
  ead_ccf    numeric,                    -- credit conversion factor for undrawn amounts; null = unchanged
  source     text    not null,           -- where the number came from
  notes      text,
  primary key (scenario, segment)
);

-- One row per source document. Licence facts live here so every chunk inherits them.
create table if not exists kb_documents (
  doc_id          text primary key,      -- 'asc326', 'ips_acl_2020'
  title           text not null,
  issuer          text not null,         -- 'FASB', 'OCC/FRB/FDIC/NCUA'
  standard        text not null check (standard in ('ASC326', 'IFRS9', 'GUIDANCE')),
  authority       text not null check (authority in ('authoritative', 'interpretive', 'supervisory')),
  framework       text,                  -- 'CECL', 'IFRS9', 'capital', 'model_risk'
  edition         text,                  -- codification update / publication date as loaded
  effective_from  date,
  url             text,
  licence         text not null,
  redistributable boolean not null,      -- false = never export into published training data
  source_file     text,
  content_hash    text,
  loaded_at       timestamptz not null default now()
);

alter table kb_documents add column if not exists framework text;  -- for databases created before it existed

-- One row per paragraph (or small group of paragraphs) for retrieval.
create table if not exists standard_chunks (
  id              bigserial primary key,
  doc_id          text not null references kb_documents (doc_id) on delete cascade,
  standard        text not null check (standard in ('ASC326', 'IFRS9', 'GUIDANCE')),
  paragraph_ref   text not null,         -- '326-20-30-2' or 'IPS Reversion 1': cited verbatim
  heading         text,
  section_path    text,                  -- '326 > 20 Amortized cost > 30 Initial measurement'
  doc_part        text not null default 'main_text' check (doc_part in (
                    'main_text', 'implementation_guidance', 'glossary', 'transition',
                    'basis_for_conclusions', 'illustrative_example', 'supervisory_guidance')),
  ordinal         integer not null,      -- reading order within the document, for neighbour lookups
  body            text not null,
  topics          text[] not null default '{}',  -- 'reversion', 'pcd', 'qualitative_factors', ...
  cross_refs      text[] not null default '{}',  -- ASC paragraphs this chunk cites
  defined_terms   text[] not null default '{}',  -- glossary terms used in the body
  effective_from  date,
  superseded      boolean not null default false,
  redistributable boolean not null,      -- copied from kb_documents at load time
  token_count     integer,               -- approximate (chars / 4), for context budgeting
  content_hash    text,
  embedding       vector(768),           -- nomic-embed-text via Ollama
  tsv             tsvector generated always as (
                    setweight(to_tsvector('english', coalesce(heading, '')), 'A') ||
                    setweight(to_tsvector('english', body), 'B')) stored,
  loaded_at       timestamptz not null default now(),
  unique (doc_id, paragraph_ref)
);

create index if not exists standard_chunks_embedding_idx on standard_chunks using hnsw (embedding vector_cosine_ops);
create index if not exists standard_chunks_tsv_idx on standard_chunks using gin (tsv);
create index if not exists standard_chunks_topics_idx on standard_chunks using gin (topics);
create index if not exists standard_chunks_xref_idx on standard_chunks using gin (cross_refs);
create index if not exists standard_chunks_order_idx on standard_chunks (doc_id, ordinal);

-- Exact lookup of defined terms ("credit-impaired", "collateral-dependent financial asset").
create table if not exists kb_definitions (
  standard      text not null,
  term          text not null,
  definition    text not null,
  doc_id        text not null references kb_documents (doc_id) on delete cascade,
  paragraph_ref text,
  primary key (standard, term)
);
