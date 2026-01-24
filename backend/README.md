# Backend - GoldenDuck

This directory contains the backend services for the GoldenDuck project, featuring a PostgreSQL database and a FastAPI service for synthetic financial data generation using GARCH modeling.

## Services

- **db**: PostgreSQL 15 database for persistent storage
- **garch**: FastAPI application serving the GARCH-based synthetic data generation service
- **flyway**: Database migration tool for schema management

## Getting Started

### Prerequisites

- Docker
- Docker Compose
- Python 3.9+ (for local development)

### Running the Services

1. Navigate to the backend directory:

   ```bash
   cd backend
   source docker-compose.env
   docker-compose up -d --build
   ```

2. Start all services:

   ```bash
   docker-compose --env-file docker-compose.env up -d --build
   ```

3. Verify services are running:
   - **FastAPI Interactive Docs**: http://localhost:8000/docs
   - **Health Check**: http://localhost:8000/health
   - **Database**: PostgreSQL on `localhost:5432`
   - **Redis**:: localhost:6379

4. View logs:

   ```bash
   # All services
   docker-compose logs -f

   # GARCH service only
   docker-compose logs -f garch
   ```

## CLI Runner (Recommended)

For convenience, an end‑to‑end CLI runner script automates the GARCH workflow without manual `curl` calls.

- Submits a GARCH job to the backend
- Polls job status automatically
- Prints model parameters and validation metrics
- Downloads the generated CSV results

**Location:** `scripts/run_garch_job.py`

### Usage

Run the script from the backend directory (ensure services are up):

```bash
cd backend
python3 worker/GARCH/scripts/run_garch_job.py --ticker AAPL
```

### Defaults
- `num_scenarios`: `100`
- `horizon`: `252`
- `p`: `1`, `q`: `1`

### Custom Configuration

Parameterize the job with flags:

```bash
python3 scripts/run_garch_job.py \
  --ticker AAPL \
  --scenarios 1000 \
  --horizon 252 \
  --p 1 \
  --q 1
```

### Options

- `--ticker` (required): Symbol like `AAPL`, `GLD`, `BTC-USD`
- `--scenarios` (default: `100`): Number of paths to generate
- `--horizon` (default: `252`): Forecast horizon in days
- `--p`, `--q` (default: `1`): GARCH model orders

### Quick Test

Test the GARCH model with a simple request:

```bash
curl -X POST http://localhost:8000/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "ticker": "AAPL",
    "num_scenarios": 10,
    "horizon": 100,
    "p": 1,
    "q": 1,
    "distribution": "skewt"
  }'
```

Expected response:
```json
{
  "job_id": "uuid-here",
  "status": "processing",
  "message": "Job started successfully"
}
```

Check job status:
```bash
curl http://localhost:8000/api/status/{job_id}
```


## Evaluation Runner

Run controlled evaluation experiments across multiple distributions and tickers.

- Evaluates `normal`, `t`, and `skewt` distributions
- Prints parameters and validation metrics per ticker
- Useful for quick model comparison and sanity checks

**Location (host):** `GARCH/scripts/run_evaluation.py`  
**Location (container):** `scripts/run_evaluation.py`  
**Tickers:** Configure in `GARCH/config/tickers.py` (`SUPPORTED_TICKERS`)

### Run (inside container)

```bash
docker-compose exec worker python GARCH/scripts/run_evaluation.py
```

### What It Does

- Downloads historical data for each ticker in `SUPPORTED_TICKERS`
- Runs `EvaluationRunner.compare_distributions(["normal","t","skewt"])`
- Prints per‑distribution results for each ticker

### Example Output

```plaintext
=== Evaluating AAPL ===
{ 'AAPL': {
    'normal': { 'parameters': {...}, 'metrics': {...}, 'num_scenarios': 500, 'horizon': 252 },
    't':      { 'parameters': {...}, 'metrics': {...}, 'num_scenarios': 500, 'horizon': 252 },
    'skewt':  { 'parameters': {...}, 'metrics': {...}, 'num_scenarios': 500, 'horizon': 252 }
  }
}
```

### Stopping Services

```bash
docker-compose down

# To remove volumes as well (deletes database data)
docker-compose down -v
```

## GARCH Model Design

### Overview

The GARCH (Generalized Autoregressive Conditional Heteroskedasticity) service generates synthetic financial time series data that captures:
- **Volatility clustering** (periods of high/low volatility)
- **Fat tails** (extreme events more frequent than normal distribution)
- **Asymmetric returns** (negative skewness - crashes vs. rallies)

### Objectives

1. **Synthetic Data Generation**: Create realistic price scenarios for risk analysis, backtesting, and stress testing
2. **Volatility Modeling**: Capture time-varying volatility dynamics in financial markets
3. **Multi-Asset Support**: Handle stocks, commodities, FX, crypto, and other asset classes
4. **Statistical Validation**: Ensure generated data matches historical distributions

### Mathematical Formulation

#### GARCH(p,q) Model

**Return Equation:**
```
r_t = Î¼ + Îµ_t
```

**Error Term:**
```
Îµ_t = Ïƒ_t Â· z_t
```

**Variance Equation (GARCH):**
```
ÏƒÂ²_t = Ï‰ + Î£(i=1 to q) Î±_iÂ·ÎµÂ²_{t-i} + Î£(j=1 to p) Î²_jÂ·ÏƒÂ²_{t-j}
```

**For GARCH(1,1):**
```
ÏƒÂ²_t = Ï‰ + Î±Â·ÎµÂ²_{t-1} + Î²Â·ÏƒÂ²_{t-1}
```

Where:
- **Ï‰ (omega)** > 0: Constant term (baseline variance)
- **Î± (alpha)** â‰¥ 0: ARCH coefficient (shock/news impact)
- **Î² (beta)** â‰¥ 0: GARCH coefficient (volatility persistence)
- **Stationarity condition**: Î± + Î² < 1
- **z_t**: Standardized innovations ~ Distribution(Î½, Î»)

#### Distribution Options

1. **Normal Distribution**: z_t ~ N(0, 1)
   - Symmetric, thin tails
   - Best for: Low volatility assets

2. **Student-t Distribution**: z_t ~ t(Î½)
   - Fat tails controlled by degrees of freedom Î½
   - Best for: Capturing extreme events

3. **Skewed Student-t Distribution**: z_t ~ Skew-t(Î½, Î») **(Recommended)**
   - Fat tails (Î½) + asymmetry (Î»)
   - Î» < 0: Negative skew (more extreme losses)
   - Best for: Equities, most financial assets

### Data Requirements

#### Input Data
- **Source**: Yahoo Finance (via `yfinance` library)
- **Format**: OHLCV (Open, High, Low, Close, Volume) daily prices
- **Minimum Period**: 500 trading days (~2 years recommended)
- **Ticker Format**: Standard symbols (e.g., `AAPL`, `GLD`, `BTC-USD`)

#### Output Data
- **Format**: JSON or CSV
- **Structure**: 
  - Scenarios: num_scenarios Ã— horizon matrix
  - Each scenario: Daily price path
  - Metadata: Model parameters, validation metrics

### Model Architecture

#### Components

```
 1. Data Ingestion (yfinance)                       
    - Fetch historical OHLCV data                    
    - Calculate log returns                          

 2. GARCH Model Fitting (arch library)              
    - Estimate parameters (omega, alpha, beta, Î½, Î»)           
    - Distribution: Skewed Student-t                 
    - Optimization: Maximum Likelihood Estimation    


 3. Scenario Generation                              
    - Simulate volatility paths (Ïƒ_t)                
    - Sample innovations (z_t)                       
    - Generate price paths via geometric Brownian    

 4. Validation (Kolmogorov-Smirnov, ACF, Kurtosis) 
    - Statistical tests for distribution matching    
    - Volatility clustering verification             

```

#### Default Configuration

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| **p** (GARCH order) | 1 | 1-3 | Number of lagged variance terms |
| **q** (ARCH order) | 1 | 1-3 | Number of lagged squared error terms |
| **distribution** | `skewt` | `normal`, `t`, `skewt` | Innovation distribution |
| **num_scenarios** | 100 | 1-10000 | Number of price paths to generate |
| **horizon** | 252 | 1-1260 | Forecast horizon (days) |

#### Expected Parameter Ranges by Asset Class

| Asset Class | Î± (ARCH) | Î² (GARCH) | Î± + Î² | Distribution |
|-------------|----------|-----------|-------|--------------|
| **Equities** (AAPL, MSFT) | 0.05-0.20 | 0.70-0.90 | 0.90-0.98 | skewt (Î» < 0) |
| **Commodities** (GLD, Oil) | 0.15-0.35 | 0.55-0.80 | 0.80-0.95 | skewt |
| **FX** (EUR/USD) | 0.03-0.10 | 0.85-0.95 | 0.90-0.99 | t or normal |
| **Crypto** (BTC, ETH) | 0.20-0.45 | 0.45-0.70 | 0.75-0.90 | skewt (high Î½) |

### Implementation Details

#### Technology Stack

- **Library**: `arch` v6.0+ (Python GARCH modeling)
- **Distribution Fitting**: Maximum Likelihood Estimation (MLE)
- **Optimization**: SLSQP with iteration limit of 5000
- **Data Scaling**: Returns multiplied by 100 for numerical stability
- **Validation**: `scipy.stats` for KS test, ACF calculation

#### Key Services

1. **GARCHService** (`services/garch_service.py`)
   - Model fitting and parameter estimation
   - Scenario generation
   - Model persistence

2. **ValidationService** (`services/validation_service.py`)
   - Kolmogorov-Smirnov test (distribution matching)
   - Autocorrelation function (volatility clustering)
   - Kurtosis comparison (fat tail matching)

3. **ModelCache** (`services/model_cache.py`)
   - In-memory caching of fitted models
   - 24-hour TTL (time-to-live)
   - Reduces redundant computation

#### Performance Optimization

- **First request**: 15-30 seconds (data fetch + model fitting)
- **Cached requests**: 200-500ms (generation only)
- **Convergence**: Retry logic with increasing max iterations
- **Scaling**: Return normalization (Ã—100) improves convergence rate

### Validation Metrics

#### 1. Kolmogorov-Smirnov Test
- **Metric**: KS statistic and p-value
- **Target**: p-value > 0.05 (cannot reject similarity)
- **Interpretation**: Tests if synthetic return distribution matches historical

#### 2. Kurtosis
- **Metric**: Excess kurtosis
- **Target**: Synthetic kurtosis â‰ˆ historical kurtosis (Â±30%)
- **Interpretation**: Measures fat tail matching

#### 3. Autocorrelation Function (ACF)
- **Metric**: ACF at lag 1
- **Target**: Synthetic ACF â‰ˆ historical ACF (Â±20%)
- **Interpretation**: Validates volatility clustering preservation

#### Example Validation Results (AAPL with skewed-t):
```
KS p-value: 0.914 âœ… (excellent)
Kurtosis: Historical=13.76, Synthetic=18.76 âœ… (good match)
ACF lag-1: Historical=0.196, Synthetic=0.179 âœ… (good match)
```

### Integration

The GARCH model is exposed via RESTful API endpoints in the FastAPI service. See API_DOCUMENTATION.md for detailed endpoint specifications.

### Model Convergence

**Common Issues:**
- **Warning**: "GARCH model did not converge"
  - **Cause**: Optimization didn't reach stopping criteria
  - **Impact**: Parameters still usable if Î± + Î² < 1
  - **Fix**: Increase `maxiter` to 10000 or check for data outliers

**Stationarity Check:**
- Always verify Î± + Î² < 1
- If Î± + Î² â‰¥ 1: Non-stationary (infinite variance) - reject fit

### Future Enhancements

- [ ] Markov-Switching GARCH for regime changes
- [ ] GJR-GARCH for leverage effects
- [ ] EGARCH for asymmetric volatility
- [ ] Multi-asset correlation modeling (DCC-GARCH)
- [ ] Real-time model refitting scheduler
- [ ] Redis-based distributed caching

## Development

### Local Setup (Without Docker)

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
cd GARCH
pip install -r requirements.txt

# Run locally
uvicorn main:app --reload --port 8000
```

### Environment Variables

See `docker-compose.env` or `.env` for configuration options:
- `DB_HOST`, `DB_PORT`, `DB_NAME`: Database connection
- `DB_USER`, `DB_PASSWORD`: Database credentials

### Running Tests

```bash
# Inside GARCH directory
pytest tests/ -v
```

## Troubleshooting

### Service Won't Start
```bash
# Check logs
docker-compose logs garch

# Rebuild from scratch
docker-compose down -v
docker-compose up --build
```

### Database Connection Issues
```bash
# Verify database is running
docker-compose ps

# Check database logs
docker-compose logs db
```

### Model Fitting Errors
- Ensure ticker symbol is valid (check Yahoo Finance)
- Minimum 500 days of historical data required
- Check for data gaps or delisted securities


Job stuck in queued

Check worker logs:

docker-compose logs -f worker

## Support

For questions or issues:
1. Check logs: `docker-compose logs -f garch`
2. Review API docs: http://localhost:8000/docs
3. Contact: [Your team contact]

---

**Version**: 1.1  
**Last Updated**: January 21, 2026