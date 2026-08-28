-- Transcript language (copied from the episode or the transcript tag) so
-- text search can pick a language-appropriate configuration.
ALTER TABLE transcripts ADD COLUMN language TEXT NOT NULL DEFAULT '';
