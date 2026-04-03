# DDPM: Denoising Diffusion Probabilistic Models for Financial Time Series

This module implements a two-stage DDPM architecture for generating synthetic market scenarios with strong regime conditioning.

## Overview

**Stage 1: Unconditional DDPM**
- Model: Conv1D U-Net with sinusoidal time embeddings
- Task: Learn the distribution of 7-asset return sequences (1260-day windows)
- Inference: Pure noise → clean returns via iterative denoising
- Use case: Baseline scenario generation

**Stage 2: Conditional DDPM with Classifier-Free Guidance (CFG)**
- Model: ConditionalDenoiser with strong per-block conditioning injection
- Conditioning: 4D market regime vector [realised_vol, drift, tail_index, momentum]
- Task: Learn conditional distribution given market regimes (calm/high-vol/crisis)
- Inference: CFG interpolation between unconditional and conditional predictions
- Use case: Regime-aware scenario generation (VaR, stress testing)

## Architecture

### Stage 1: Conv1DUNet (Unconditional)
```
Input (B, 7, 1260)
  ↓
Encoder:
  ResBlock (→64)  + skip +
  ResBlock (→64)  + skip +
  ResBlock (→64)  + skip +
  ResBlock (→64)  + skip +
Bottleneck (→64)
Decoder (with skip concatenation):
  ResBlock (→64)  + skip
  ResBlock (→64)  + skip
  ResBlock (→64)  + skip
  ResBlock (→64)  + skip
  ↓
Output (B, 7, 1260)
```

**Key Components:**
- `SinusoidalPosEmb`: Time embeddings (128-dim) via sin/cos functions
- `ResBlock`: AdaGN (Adaptive Group Norm) with time conditioning at each layer
- Exponential dilation: 1→2→4→8→16 across encoder layers
- Skip connections: Encoder outputs concatenated in decoder

### Stage 2: ConditionalDenoiser (Conditional + CFG)
```
Input (B, 7, 1260)
Conditioning (B, 4)
  ↓
Time + Conditioning Embedding (shared)
  ↓
Encoder with per-block conditioning injection:
  enc1: Conv → conditioning scale + skip +
  enc2: Conv → conditioning scale + skip +
  enc3: Conv → conditioning scale + skip +
Bottleneck: Conv → conditioning scale
Decoder (with skip concatenation):
  dec3: dec3(concat[m, e3])
  dec2: dec2(concat[d3, e2])
  dec1: dec1(concat[d2, e1])
  ↓
Output (B, 7, 1260)

CFG Inference (at each timestep):
  ε̂ = ε_uncond + guidance_scale × (ε_cond - ε_uncond)
```

**Key Differences from Stage 1:**
- Conditioning injected at **every encoder + bottleneck block** (not just time embedding)
- Per-block `ConditioningBlock` projects time+cond → channel-wise scale factors
- Supports CFG: zero-conditioning path evaluates unconditional prediction

## Configuration

All hyperparameters are centralized in [`config/ddpm_config.py`](config/ddpm_config.py):

### Global Settings
```python
TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA']
WINDOW_LEN = 1260  # ~5 years of trading days
STRIDE = 5
NUM_ASSETS = 7
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
```

### Stage 1 Config
```python
T_STEPS = 200              # Diffusion steps
BASE_CHANNELS = 64
DEPTH = 4                  # Network depth (exponent for dilation)
EPOCHS = 150
LR = 2e-4
EMA_DECAY = 0.9999        # Exponential moving average for scheduler
PATIENCE = 30              # Early stopping
```

### Stage 2 Config
```python
COND_EPOCHS = 300
COND_LR = 1e-4
CFG_DROP_PROB = 0.10       # Prob of conditioning during training (CFG)
DEFAULT_GUIDANCE_SCALE = 3.0
T_STEPS = 200
```

### Regime Definitions
```python
REGIME_CALM = {'RV_THRESHOLD': 0.10, 'DRIFT': 0.02, 'TAIL': 0.02}
REGIME_HIGHVOL = {'RV_THRESHOLD': 0.30, 'DRIFT': 0.00, 'TAIL': 0.035}
REGIME_CRISIS = {'RV_THRESHOLD': 0.55, 'DRIFT': -0.05, 'TAIL': 0.05}
```

## Module Structure

```
backend/worker/DDPM/
├── config/
│   ├── __init__.py
│   └── ddpm_config.py          # All hyperparameters
├── models/
│   ├── __init__.py
│   ├── schedulers.py           # DDPMScheduler (noise schedule)
│   ├── denoiser.py             # Conv1DUNet (Stage 1)
│   ├── conditional_denoiser.py # ConditionalDenoiser (Stage 2)
│   └── schemas.py              # Pydantic models
├── services/
│   ├── __init__.py
│   ├── data_processor.py       # Download, returns, windowing, scaling
│   ├── conditioning.py         # Market regime vectors
│   ├── inference.py            # UnconditionalInferenceService
│   ├── validation.py           # DDPMValidationService
│   └── ddpm_service.py         # High-level orchestration
├── scripts/
│   ├── __init__.py
│   └── test_inference.py       # Inference tests
├── utils/
│   └── __init__.py
├── tests/
│   └── __init__.py
├── README.md                   # This file
└── requirements.txt            # Dependencies
```

## Data Processing Pipeline

### 1. Download Historical Data
```python
processor = DataProcessor(device='cuda')
data = processor.download_data(
    tickers=['AAPL', 'MSFT', ...],
    start_date='2010-01-01',
    end_date='2024-12-31'
)
```

### 2. Compute Log Returns
```python
returns = processor.compute_returns(data, winsorize=True)
# Per-asset winsorization at ±10σ to remove outliers
```

### 3. Create Rolling Windows
```python
windows = processor.create_rolling_windows(
    returns,
    window_len=1260,
    stride=5
)
# Shape: (num_windows, 7, 1260)
```

### 4. Standardize Globally
```python
standardized, (mean, std) = processor.global_standardize(windows, fit=True)
# (X - global_mean) / global_std across all assets and time
```

### 5. Compute Stress Scores
```python
stress_scores = processor.stress_scoring(standardized)
# Exponential-weighted volatility (higher weight on recent observations)
```

## Market Regime Conditioning

Conditioning vector (4D) captures market regimes:
```python
[realised_vol, drift, tail_index, momentum]
```

### Computation Example
```python
from services.conditioning import compute_conditioning_from_window, detect_regime

# From a single rolling window
cond_vec = compute_conditioning_from_window(window)
# realised_vol: 0.25 (annualised)
# drift: 0.05 (annualised)
# tail_index: 0.03 (99th percentile of |returns|)
# momentum: 0.15 (ACF lag-1)

# Detect regime
regime = detect_regime(cond_vec, vol_threshold_low=0.25, vol_threshold_high=0.50)
# Returns: 'calm' | 'highvol' | 'crisis'
```

## Inference API

### Stage 1: Unconditional Generation
```python
from models.schedulers import DDPMScheduler
from models.denoiser import Conv1DUNet
from services.inference import InferenceService

model = Conv1DUNet(num_assets=7, base_channels=64, depth=4)
scheduler = DDPMScheduler(T=200, device='cuda')
inference = InferenceService(device='cuda')

paths = inference.generate_stage1(
    model=model,
    scheduler=scheduler,
    num_paths=2000,
    seed=42
)
# Output: (2000, 7, 1260)
```

### Stage 2: Conditional with CFG
```python
from models.conditional_denoiser import ConditionalDenoiser

model = ConditionalDenoiser(
    seq_len=1260,
    in_ch=7,
    cond_dim=4,
    base_ch=64
)
scheduler = DDPMScheduler(T=200, device='cuda')

conditioning = np.array([0.30, 0.02, 0.04, -0.10])  # High-vol regime

paths = inference.generate_stage2_cfg(
    model=model,
    scheduler=scheduler,
    conditioning=conditioning,
    num_paths=2000,
    guidance_scale=3.0,  # Strength of conditioning
    seed=42
)
# Output: (2000, 7, 1260)
```

### Unified Interface
```python
# Works for both stages
paths = inference.generate_paths(
    model=model,
    scheduler=scheduler,
    stage=2,
    num_paths=2000,
    conditioning=conditioning,
    guidance_scale=3.0,
    seed=42
)
```

## Classifier-Free Guidance (CFG)

**Training (Stage 2):**
- Randomly drop conditioning with probability `CFG_DROP_PROB` (default 10%)
- Model learns both conditional and unconditional predictions

**Inference:**
```
ε̂(x_t, t, c) = ε_uncond(x_t, t) + guidance_scale × (ε_cond(x_t, t, c) - ε_uncond(x_t, t))
```

**Effect of guidance_scale:**
- 1.0: No guidance (purely unconditional)
- 3.0: Moderate conditioning strength (default)
- 10.0: Strong conditioning (may reduce diversity)

## Validation Metrics

### Standard Statistical Metrics
- **KS Test**: Kolmogorov-Smirnov test (p-value > 0.05 is good)
- **Kurtosis**: Excess kurtosis (should match historical)
- **Skewness**: Return distribution skewness
- **ACF**: Autocorrelation at lag-1 (momentum)
- **ACF of squared returns**: Volatility clustering

### DDPM-Specific Metrics (Stage 2)
- **Terminal return ratio**: Cumulative return at end of path
- **Rolling volatility ratio**: Quarterly volatility statistics
- **Regime hit rates**: Distribution across calm/high-vol/crisis regimes
  - Target: 40% calm, 45% high-vol, 15% crisis

### Quality Assessment
```python
from services.validation import DDPMValidationService

validation = DDPMValidationService()
metrics = validation.validate_scenarios(
    historical_data=hist_returns,
    synthetic_data=synth_returns,
    stage=2,
    regime_segment_len=63
)

quality = validation.assess_quality(metrics)
# Returns: 'PASS' | 'FAIL'
```

## High-Level Orchestration

```python
from services import DDPMService

service = DDPMService(stage=2, device='cuda')
service.initialize(
    data_processor=processor,
    scheduler=scheduler,
    model=model,
    inference_service=inference,
    validation_service=validation
)

# Step 1: Fit (training metadata)
training_meta = service.fit_with_retry(training_data)

# Step 2: Generate paths
paths, gen_meta = service.generate_scenarios(
    num_paths=2000,
    conditioning=cond_vec,
    guidance_scale=3.0
)

# Step 3: Validate
metrics = service.validate_scenarios(
    historical_data=hist_data,
    synthetic_data=paths
)

# Summary
summary = service.get_summary()
print(summary)
```

## Testing

Run comprehensive inference tests:
```bash
python scripts/test_inference.py
```

Tests include:
- Stage 1 unconditional generation
- Stage 2 conditional generation with different guidance scales
- Unified interface routing
- Multi-batch generation
- Seed-based reproducibility

## Dependencies

See [`requirements.txt`](requirements.txt) for full dependency list.

**Key libraries:**
- `torch >= 2.0`: Deep learning framework
- `numpy >= 1.24`: Numerical computing
- `pandas >= 2.0`: Data manipulation
- `scipy >= 1.10`: Statistical functions
- `scikit-learn >= 1.3`: Machine learning utilities
- `yfinance >= 0.2.28`: Financial data download
- `pydantic >= 2.0`: Data validation

## Usage Examples

### Example 1: Generate Scenarios for VaR Calculation
```python
# Assume model and scheduler are trained
service = DDPMService(stage=2, device='cuda')

# High-volatility regime
cond_crisis = np.array([0.50, -0.03, 0.05, -0.20])
paths_crisis, _ = service.generate_scenarios(
    num_paths=10000,
    conditioning=cond_crisis,
    guidance_scale=5.0
)

# Compute terminal returns
terminal_returns = np.exp(paths_crisis.sum(axis=2).mean(axis=1))
var_95 = np.percentile(terminal_returns, 5)
```

### Example 2: Stress Testing
```python
# Calm regime
cond_calm = np.array([0.15, 0.03, 0.02, 0.10])
paths_calm, _ = service.generate_scenarios(
    num_paths=5000,
    conditioning=cond_calm,
    guidance_scale=3.0
)

# Peak-to-trough drawdown
def max_drawdown(returns):
    cumsum = np.cumprod(1 + returns.mean(axis=1))
    running_max = np.maximum.accumulate(cumsum)
    return (1 - cumsum / running_max).max()

mdd = max_drawdown(paths_calm)
```

## Performance Notes

- **Memory**: Stage 1 requires ~8GB GPU memory for batch_size=32 at 1260-length sequences
- **Inference speed**: ~2000 paths/min on single V100 (200 denoising steps)
- **Convergence**: Stage 1 typically converges in 50-100 epochs; Stage 2 requires 200-300 epochs

## References

1. **DDPM**: Ho et al. (2020) "Denoising Diffusion Probabilistic Models"
2. **Classifier-Free Guidance**: Ho & Salimans (2021) "Classifier-Free Diffusion Guidance"
3. **Diffusion for Time Series**: Rasul et al. (2021) "Autoregressive Denoising Diffusion Models"

