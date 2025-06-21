CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

CREATE TABLE dokument (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    blob_path       VARCHAR NOT NULL,
    mime_type       VARCHAR NOT NULL,
    uploaded_at     TIMESTAMP DEFAULT now(),
    hash_sha256     CHAR(64) NOT NULL,
    source_filename VARCHAR,
    document_type   VARCHAR,
    linked_entity   VARCHAR,
    linked_entity_id VARCHAR,
    textextraction_status VARCHAR DEFAULT 'not ready' CHECK (textextraction_status IN ('not ready', 'ready', 'in progress', 'done', 'error'))
);

CREATE TABLE extraction_job (
    id            UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    dokument_id   UUID REFERENCES dokument(id) ON DELETE CASCADE,
    created_at    TIMESTAMP DEFAULT now(),
    finished_at   TIMESTAMP,
    state         VARCHAR,
    worker_log    TEXT
); 