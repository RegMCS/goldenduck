-- Manually uploaded model artifacts (versioned .pkl in S3, same naming as training worker)
CREATE TABLE IF NOT EXISTS uploaded_models (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_name TEXT NOT NULL UNIQUE,
    s3_key TEXT NOT NULL,
    uploaded_by UUID NOT NULL REFERENCES users (id),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    file_size_bytes BIGINT,
    is_active BOOLEAN NOT NULL DEFAULT FALSE
);

CREATE INDEX IF NOT EXISTS idx_uploaded_models_created_at ON uploaded_models (created_at DESC);
