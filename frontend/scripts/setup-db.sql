-- Users table (synced with Firebase Auth)
CREATE TABLE IF NOT EXISTS users (
  id SERIAL PRIMARY KEY,
  firebase_uid VARCHAR(128) UNIQUE NOT NULL,
  email VARCHAR(255) UNIQUE NOT NULL,
  display_name VARCHAR(255),
  created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Generation history table
CREATE TABLE IF NOT EXISTS generation_history (
  id SERIAL PRIMARY KEY,
  user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
  scenario_name VARCHAR(255),
  asset_class VARCHAR(50) NOT NULL,
  trend_type VARCHAR(50) NOT NULL,
  market_regime VARCHAR(50) NOT NULL,
  volatility_level VARCHAR(50) NOT NULL,
  generation_model VARCHAR(50) NOT NULL,
  time_horizon INTEGER NOT NULL,
  data_points INTEGER NOT NULL,
  correlation_strength INTEGER NOT NULL,
  mean_reversion BOOLEAN DEFAULT FALSE,
  fat_tails BOOLEAN DEFAULT FALSE,
  jump_diffusion BOOLEAN DEFAULT FALSE,
  config_json JSONB,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Index for faster queries
CREATE INDEX IF NOT EXISTS idx_generation_history_user_id ON generation_history(user_id);
CREATE INDEX IF NOT EXISTS idx_generation_history_created_at ON generation_history(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_users_firebase_uid ON users(firebase_uid);
