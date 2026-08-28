CREATE TABLE jobs (
    id               BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    job_type         TEXT        NOT NULL,
    payload          JSONB       NOT NULL DEFAULT '{}',
    idempotency_key  TEXT,
    status           TEXT        NOT NULL DEFAULT 'pending'
                     CHECK (status IN ('pending', 'running', 'completed', 'dead')),
    run_at           TIMESTAMPTZ NOT NULL DEFAULT now(),
    attempts         INTEGER     NOT NULL DEFAULT 0,
    max_attempts     INTEGER     NOT NULL DEFAULT 5,
    lease_expires_at TIMESTAMPTZ,
    last_error       TEXT        NOT NULL DEFAULT '',
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    completed_at     TIMESTAMPTZ
);

-- Re-enqueueing the same logical job is a no-op.
CREATE UNIQUE INDEX jobs_idempotency_key ON jobs (job_type, idempotency_key)
    WHERE idempotency_key IS NOT NULL;

-- Claim query: pending jobs that are due, oldest first.
CREATE INDEX jobs_claim_idx ON jobs (job_type, run_at)
    WHERE status = 'pending';

-- Lease reaper: running jobs whose lease may have expired.
CREATE INDEX jobs_lease_idx ON jobs (lease_expires_at)
    WHERE status = 'running';
