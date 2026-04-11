-- Migrate TIMESTAMP columns to TIMESTAMPTZ to preserve SGT (+08:00) offset
ALTER TABLE ai_model_jobs
    ALTER COLUMN requested_at TYPE TIMESTAMPTZ USING requested_at AT TIME ZONE 'UTC',
    ALTER COLUMN completed_at TYPE TIMESTAMPTZ USING completed_at AT TIME ZONE 'UTC';
