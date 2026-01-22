-- Create users table first
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    first_name TEXT,
    last_name TEXT
);

-- Create ENUM type for status and job_type
DO $$ BEGIN
    CREATE TYPE job_status AS ENUM ('queued', 'running', 'completed', 'failed');
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

DO $$ BEGIN
    CREATE TYPE job_type AS ENUM ('garch', 'gan', 'ddpm');
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

-- Create table ai_model_jobs
CREATE TABLE IF NOT EXISTS ai_model_jobs (
    id UUID PRIMARY KEY,
    status job_status NOT NULL,
    job_type job_type,
    requestor UUID NOT NULL,
    requested_at TIMESTAMP NOT NULL,
    s3_url TEXT,
    completed_at TIMESTAMP,
    CONSTRAINT fk_user FOREIGN KEY (requestor) REFERENCES users(id)
);
