"""
DDPM Configuration - All hyperparameters for Stage 1 and Stage 2
"""

# ═══════════════════════════════════════════════════════════════════════════════
# GLOBAL SETTINGS
# ═══════════════════════════════════════════════════════════════════════════════

DEVICE = "cuda"  # Will be auto-detected if cuda available
TICKERS = ["^GSPC", "AAPL", "CL=F", "GC=F", "EURUSD=X", "BTC-USD", "^VIX"]
NUM_ASSETS = len(TICKERS)
START_DATE = "2010-01-01"
END_DATE = "2025-12-31"

# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 1: UNCONDITIONAL DDPM
# ═══════════════════════════════════════════════════════════════════════════════

# Windowing
WINDOW_LEN = 1260   # ~5 trading years
STRIDE = 5

# Diffusion
T_STEPS = 200

# Model Architecture
BASE_CHANNELS = 64
TIME_EMB_DIM = 128
DEPTH = 4

# Training
BATCH_SIZE = 64
EPOCHS = 150
LR = 2e-4
PATIENCE = 30
EMA_DECAY = 0.9999
WEIGHT_DECAY = 1e-4

# Stress Weighting
STRESS_ALPHA = 6.0

# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 2: CONDITIONAL DDPM (USER KNOBS)
# ═══════════════════════════════════════════════════════════════════════════════

# Conditioning Vector Dimension
CONDITIONING_DIM = 4  # [realised_vol, drift, tail_index, momentum]

# Conditioning Normalization Params (computed during training)
CONDITIONING_NORM_PARAMS = {
    'min': None,  # Will be computed from training data
    'max': None
}

# Conditional Model Architecture
COND_BASE_CHANNELS = 64
COND_TIME_EMB_DIM = 64
COND_EMB_DIM = 64

# Conditional Training
COND_BATCH_SIZE = 32
COND_EPOCHS = 300
COND_LR = 1e-4
COND_WEIGHT_DECAY = 1e-5
COND_PATIENCE = 20

# Classifier-Free Guidance (CFG)
CFG_DROP_PROB = 0.10  # 10% of batches drop conditioning
DEFAULT_GUIDANCE_SCALE = 3.0

# Tail-weighted loss
TAIL_WEIGHT_THRESHOLD = 0.5  # In normalised units
TAIL_WEIGHT_MULTIPLIER = 3.0

# ═══════════════════════════════════════════════════════════════════════════════
# GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

NUM_PATHS = 2000
GEN_BATCH = 256

# De-normalization (will be computed from training data)
WINDOW_SCALES_SAVED = None  # (num_windows, num_assets) - median used for generation

# ═══════════════════════════════════════════════════════════════════════════════
# DATA PREPROCESSING
# ═══════════════════════════════════════════════════════════════════════════════

# Winsorization
RETURN_CLIP_SIGMA = 10.0  # Clip returns at ±10σ per asset

# Final clipping
FINAL_CLIP_MIN = -0.05
FINAL_CLIP_MAX = 0.05

# ═══════════════════════════════════════════════════════════════════════════════
# REGIMES FOR EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════

REGIME_DEFINITIONS = {
    "calm": {
        "realised_vol": 0.10,
        "drift": 0.05,
        "tail_index": 0.008,
        "momentum": -0.05
    },
    "highvol": {
        "realised_vol": 0.30,
        "drift": -0.05,
        "tail_index": 0.025,
        "momentum": -0.10
    },
    "crisis": {
        "realised_vol": 0.55,
        "drift": -0.25,
        "tail_index": 0.055,
        "momentum": -0.15
    }
}

PATHS_PER_REGIME = 200

# ═══════════════════════════════════════════════════════════════════════════════
# EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════

TRADING_DAYS = 252
ROLLING_VOL_WINDOW = 21

# Regime hit rate segmentation
REGIME_HIT_SEGMENT_LEN = 63  # ~3 months
VOL_THRESHOLD_LOW = 0.25     # Switch from calm to high-vol
VOL_THRESHOLD_HIGH = 0.50    # Switch from high-vol to crisis

# ═══════════════════════════════════════════════════════════════════════════════
# MODEL CHECKPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

CHECKPOINT_DIR = "checkpoints"
STAGE1_CHECKPOINT = f"{CHECKPOINT_DIR}/ddpm_stage1_ema.pt"
STAGE2_CHECKPOINT = f"{CHECKPOINT_DIR}/ddpm_stage2.pt"
TRAINING_ARTIFACTS = f"{CHECKPOINT_DIR}/training_artifacts.pkl"
