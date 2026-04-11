-- Add is_admin flag to users table
ALTER TABLE users
    ADD COLUMN IF NOT EXISTS is_admin BOOLEAN NOT NULL DEFAULT FALSE;

-- Create training_jobs table
CREATE TABLE IF NOT EXISTS training_jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    status VARCHAR(20) NOT NULL DEFAULT 'queued',
    triggered_by UUID NOT NULL REFERENCES users(id),
    started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMPTZ,
    -- JSON config: testing_mode, n_assets, n_scenarios, run_evaluation
    config JSONB NOT NULL DEFAULT '{}',
    -- Coarse step progress: e.g. "Step 2 of 4: Generating samples"
    step TEXT,
    step_index INTEGER,
    -- Saved model filename e.g. rf_delta_20260402_143000.pkl
    model_name TEXT,
    -- Whether this run's model is currently the active inference model
    is_active BOOLEAN NOT NULL DEFAULT FALSE,
    -- Full evaluation_report.json contents stored inline
    evaluation_report JSONB,
    error TEXT
);
