-- How a search was entered: typed, spoken (with the recognizer's transcript,
-- language, confidence and backend), or a spoken transcript the user edited
-- before searching (linked to the voice search it came from). Voice
-- recordings are discarded after transcription unless the user asked to
-- keep one; a kept recording is an object in the store, referenced here
-- with the time of consent, and deleted with the session's data.
ALTER TABLE searches ADD COLUMN input JSONB NOT NULL DEFAULT '{"mode": "text"}';

CREATE TABLE voice_recordings (
    query_id      TEXT             PRIMARY KEY REFERENCES searches (query_id) ON DELETE CASCADE,
    session_id    TEXT             NOT NULL,
    object_key    TEXT             NOT NULL,
    content_type  TEXT             NOT NULL,
    bytes         INTEGER          NOT NULL,
    audio_seconds DOUBLE PRECISION NOT NULL,
    consented_at  TIMESTAMPTZ      NOT NULL,
    created_at    TIMESTAMPTZ      NOT NULL DEFAULT now()
);

CREATE INDEX voice_recordings_session_idx ON voice_recordings (session_id, created_at);
