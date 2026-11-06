-- Explicit, additive initialization. The application does not auto-migrate.
BEGIN;
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS rag_papers (
    paper_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    source_path TEXT NOT NULL,
    source_sha256 TEXT NOT NULL CHECK (source_sha256 ~ '^[a-f0-9]{64}$'),
    embedding_model TEXT NOT NULL,
    embedding_revision TEXT NOT NULL,
    embedding_dimensions INTEGER NOT NULL CHECK (embedding_dimensions = 1024),
    query_instruction TEXT NOT NULL,
    representation_version TEXT NOT NULL,
    cleaner_version TEXT NOT NULL,
    splitter_version TEXT NOT NULL,
    chunk_size_chars INTEGER NOT NULL CHECK (chunk_size_chars > 0),
    overlap_chars INTEGER NOT NULL CHECK (overlap_chars >= 0 AND overlap_chars < chunk_size_chars),
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS rag_pages (
    paper_id TEXT NOT NULL REFERENCES rag_papers(paper_id) ON DELETE CASCADE,
    page_number INTEGER NOT NULL CHECK (page_number > 0),
    raw_text BYTEA NOT NULL,
    text TEXT NOT NULL,
    PRIMARY KEY (paper_id, page_number)
);

CREATE TABLE IF NOT EXISTS rag_chunks (
    paper_id TEXT NOT NULL,
    page_number INTEGER NOT NULL,
    chunk_index INTEGER NOT NULL CHECK (chunk_index >= 0),
    text TEXT NOT NULL CHECK (length(text) > 0),
    start_char INTEGER NOT NULL CHECK (start_char >= 0),
    end_char INTEGER NOT NULL CHECK (end_char > start_char),
    raw_start_char INTEGER NOT NULL CHECK (raw_start_char >= 0),
    raw_end_char INTEGER NOT NULL CHECK (raw_end_char > raw_start_char),
    embedding vector(1024) NOT NULL CHECK (vector_norm(embedding) > 0),
    PRIMARY KEY (paper_id, page_number, chunk_index),
    FOREIGN KEY (paper_id, page_number) REFERENCES rag_pages(paper_id, page_number) ON DELETE CASCADE,
    CHECK (length(text) = end_char - start_char)
);
COMMIT;
