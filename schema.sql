CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

CREATE TABLE Dokument (
    dokument_id        UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    dokumententyp      VARCHAR NOT NULL,
    pfad_dms           VARCHAR NOT NULL,
    erstellt_am        TIMESTAMP DEFAULT now(),
    hash_sha256        CHAR(64) NOT NULL,
    quelle_dateiname   VARCHAR,
    verknuepfte_entitaet VARCHAR,
    verknuepfte_entitaet_id VARCHAR,
    textextraktion_status VARCHAR DEFAULT 'nicht bereit' CHECK (textextraktion_status IN ('nicht bereit', 'bereit', 'in Bearbeitung', 'abgeschlossen', 'fehlerhaft'))
);

CREATE TABLE Extraktionsauftrag (
    auftrag_id         UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    dokument_id        UUID REFERENCES Dokument(dokument_id) ON DELETE CASCADE,
    status             VARCHAR DEFAULT 'Extraktion ausstehend',
    fehlermeldung      TEXT,
    erstellt_am        TIMESTAMP DEFAULT now(),
    abgeschlossen_am   TIMESTAMP
);

CREATE TABLE ExtrahierteDaten (
    id                 UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    dokument_id        UUID REFERENCES Dokument(dokument_id) ON DELETE CASCADE,
    feldname           VARCHAR NOT NULL,
    wert               TEXT,
    position_im_dokument JSONB,
    konfidenzscore     DECIMAL(5,4),
    extrahiert_am      TIMESTAMP DEFAULT now()
);

-- Create indexes for better performance
CREATE INDEX idx_extraktionsauftrag_dokument_id ON Extraktionsauftrag(dokument_id);
CREATE INDEX idx_extraktionsauftrag_status ON Extraktionsauftrag(status);
CREATE INDEX idx_extrahierte_daten_dokument_id ON ExtrahierteDaten(dokument_id);
CREATE INDEX idx_extrahierte_daten_feldname ON ExtrahierteDaten(feldname); 