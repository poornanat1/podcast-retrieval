-- Provider-reported global popularity percentile in [0, 1]; 0 when the
-- discovery provider reported none. Drives popularity baselines until
-- first-party interaction data exists.
ALTER TABLE podcasts ADD COLUMN popularity DOUBLE PRECISION NOT NULL DEFAULT 0;
CREATE INDEX podcasts_popularity_idx ON podcasts (popularity DESC);
