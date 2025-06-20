CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

CREATE TABLE dokument (
    id            UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    blob_path     VARCHAR NOT NULL,
    mime_type     VARCHAR NOT NULL,
    uploaded_at   TIMESTAMP DEFAULT now(),
    ocr_status    VARCHAR DEFAULT 'PENDING'
);

CREATE TABLE extraktionsauftrag (
    id            UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    dokument_id   UUID REFERENCES dokument(id) ON DELETE CASCADE,
    created_at    TIMESTAMP DEFAULT now(),
    finished_at   TIMESTAMP,
    state         VARCHAR
); 