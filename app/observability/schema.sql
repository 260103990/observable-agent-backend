-- Existing TraceStore contract, for explicit initialization of a new database.
BEGIN;
CREATE TABLE IF NOT EXISTS traces (
    trace_id TEXT PRIMARY KEY,
    request_id TEXT NOT NULL,
    status TEXT NOT NULL,
    duration_ms DOUBLE PRECISION NOT NULL
);
CREATE TABLE IF NOT EXISTS spans (
    trace_id TEXT NOT NULL REFERENCES traces(trace_id) ON DELETE CASCADE,
    span_index INTEGER NOT NULL,
    span_id TEXT NOT NULL,
    parent_span_id TEXT,
    name TEXT NOT NULL,
    status TEXT NOT NULL,
    duration_ms DOUBLE PRECISION NOT NULL,
    attributes JSONB NOT NULL,
    error_type TEXT,
    PRIMARY KEY (trace_id, span_index)
);
COMMIT;
