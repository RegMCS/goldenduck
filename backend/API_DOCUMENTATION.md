# GARCH Service API Documentation

## Base URL
```
http://localhost:8000
```

## Authentication
Currently no authentication required (development mode).

---

## Endpoints

### 1. Health Check

Check if the service is running.

**Endpoint:** `GET /health`

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2026-01-14T15:30:00.123456",
  "version": "1.0.0"
}
```

---

### 2. Generate Synthetic Data

Fit GARCH model and generate synthetic price scenarios.

**Endpoint:** `POST /api/generate`

**Request Body:**
```json
{
  "ticker": "AAPL",
  "num_scenarios": 100,
  "horizon": 252,
  "p": 1,
  "q": 1,
  "distribution": "skewt"
}
```

**Parameters:**

| Field | Type | Required | Default | Range | Description |
|-------|------|----------|---------|-------|-------------|
| `ticker` | string | âœ… Yes | - | - | Yahoo Finance ticker symbol (e.g., "AAPL", "GLD", "BTC-USD") |
| `num_scenarios` | integer | No | 100 | 1-10000 | Number of synthetic price paths to generate |
| `horizon` | integer | No | 252 | 1-1260 | Forecast horizon in trading days (252 = 1 year) |
| `p` | integer | No | 1 | 1-3 | GARCH order (lagged variance terms) |
| `q` | integer | No | 1 | 1-3 | ARCH order (lagged squared error terms) |
| `distribution` | string | No | "skewt" | "normal", "t", "skewt" | Innovation distribution type |

**Distribution Options:**

- `"normal"`: Normal distribution (thin tails, symmetric)
- `"t"`: Student-t distribution (fat tails, symmetric)
- `"skewt"`: Skewed Student-t distribution (fat tails + asymmetry) **[Recommended for stocks]**

**Response (202 Accepted):**
```json
{
  "job_id": "f02bcc43-a76a-4426-9b78-cf45d622f06d",
  "status": "processing",
  "message": "Job started successfully"
}
```

**Example cURL:**
```bash
curl -X POST http://localhost:8000/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "ticker": "AAPL",
    "num_scenarios": 100,
    "horizon": 252,
    "p": 1,
    "q": 1,
    "distribution": "skewt"
  }'
```

**Example Python:**
```python
import requests

response = requests.post(
    "http://localhost:8000/api/generate",
    json={
        "ticker": "AAPL",
        "num_scenarios": 100,
        "horizon": 252,
        "p": 1,
        "q": 1,
        "distribution": "skewt"
    }
)

job = response.json()
print(f"Job ID: {job['job_id']}")
```

---

### 3. Check Job Status

Poll for job completion and retrieve results.

**Endpoint:** `GET /api/status/{job_id}`

**Path Parameters:**
- `job_id` (string, required): Job ID returned from `/api/generate`

**Response (Processing):**
```json
{
  "status": "processing",
  "progress": 50
}
```

**Response (Completed):**
```json
{
  "status": "completed",
  "job_id": "f02bcc43-a76a-4426-9b78-cf45d622f06d",
  "ticker": "AAPL",
  "garch_params": {
    "omega": 1.852,
    "alpha": 0.424,
    "beta": 0.116,
    "converged": true,
    "aic": 1799.21,
    "bic": 1820.28,
    "nu": 4.23,
    "lambda": -0.15
  },
  "validation_metrics": {
    "ks_statistic": 0.030,
    "ks_pvalue": 0.914,
    "kurtosis_historical": 13.76,
    "kurtosis_synthetic": 18.76,
    "acf_lag1_historical": 0.196,
    "acf_lag1_synthetic": 0.179
  },
  "scenarios": [
    [230.15, 231.22, 229.87, ...],
    [230.15, 228.94, 227.55, ...],
    ...
  ],
  "download_url": "/api/download/f02bcc43-a76a-4426-9b78-cf45d622f06d.csv"
}
```

**Response (Failed):**
```json
{
  "status": "failed",
  "error": "Invalid ticker symbol or insufficient data"
}
```

**Field Descriptions:**

**GARCH Parameters:**
- `omega` (Ï‰): Constant baseline variance
- `alpha` (Î±): ARCH coefficient - shock sensitivity
- `beta` (Î²): GARCH coefficient - volatility persistence
- `converged`: Whether optimization converged successfully
- `aic`: Akaike Information Criterion (lower is better)
- `bic`: Bayesian Information Criterion (lower is better)
- `nu` (Î½): Degrees of freedom (Student-t) - controls tail thickness
- `lambda` (Î»): Skewness parameter (skewed-t) - negative = left skew

**Validation Metrics:**
- `ks_statistic`: Kolmogorov-Smirnov statistic (0-1, lower is better)
- `ks_pvalue`: KS test p-value (>0.05 = good match)
- `kurtosis_historical`: Excess kurtosis of historical returns
- `kurtosis_synthetic`: Excess kurtosis of synthetic returns
- `acf_lag1_historical`: ACF at lag 1 for historical returns
- `acf_lag1_synthetic`: ACF at lag 1 for synthetic returns

**Example cURL:**
```bash
curl http://localhost:8000/api/status/f02bcc43-a76a-4426-9b78-cf45d622f06d
```

**Example Python (with polling):**
```python
import requests
import time

job_id = "f02bcc43-a76a-4426-9b78-cf45d622f06d"

while True:
    response = requests.get(f"http://localhost:8000/api/status/{job_id}")
    result = response.json()
    
    if result["status"] == "completed":
        print("Job completed!")
        print(f"GARCH params: {result['garch_params']}")
        print(f"Generated {len(result['scenarios'])} scenarios")
        break
    elif result["status"] == "failed":
        print(f"Job failed: {result['error']}")
        break
    else:
        print(f"Status: {result['status']}, Progress: {result.get('progress', 0)}%")
        time.sleep(2)  # Poll every 2 seconds
```

---

### 4. Download Results (CSV)

Download generated scenarios as CSV file.

**Endpoint:** `GET /api/download/{job_id}.csv`

**Path Parameters:**
- `job_id` (string, required): Job ID

**Response:**
CSV file with format:
```csv
scenario_id,day,price
0,0,230.15
0,1,231.22
0,2,229.87
1,0,230.15
1,1,228.94
...
```

**Example cURL:**
```bash
curl http://localhost:8000/api/download/f02bcc43-a76a-4426-9b78-cf45d622f06d.csv -o scenarios.csv
```

**Example Python:**
```python
import requests
import pandas as pd

job_id = "f02bcc43-a76a-4426-9b78-cf45d622f06d"
url = f"http://localhost:8000/api/download/{job_id}.csv"

# Download and load into pandas
df = pd.read_csv(url)
print(df.head())

# Reshape to wide format (scenarios as columns)
wide_df = df.pivot(index='day', columns='scenario_id', values='price')
print(wide_df.head())
```

---

## Error Codes

| Status Code | Meaning | Example Response |
|-------------|---------|------------------|
| 200 | Success | Job status retrieved |
| 202 | Accepted | Job started |
| 400 | Bad Request | Invalid parameters |
| 404 | Not Found | Job ID not found |
| 500 | Internal Server Error | Model fitting failed |

**Error Response Format:**
```json
{
  "detail": "Error message describing the issue"
}
```

---

## Interpreting Results

### Model Quality Assessment

**1. Convergence**
- âœ… `converged: true` - Model fitted successfully
- âš ï¸ `converged: false` - Check if Î± + Î² < 1 (stationarity)

**2. AIC/BIC (Model Comparison)**
- **Lower is better** - only meaningful when comparing models
- BIC > AIC is normal (BIC penalizes complexity more)
- Improvement of 50+ points = significant improvement

**3. KS Test (Distribution Matching)**
- âœ… `ks_pvalue > 0.05` - Excellent (cannot reject similarity)
- âš ï¸ `ks_pvalue 0.01-0.05` - Marginal (acceptable for PoC)
- âŒ `ks_pvalue < 0.01` - Poor fit (consider different distribution)

**4. Kurtosis (Fat Tail Matching)**
- Target: `kurtosis_synthetic â‰ˆ kurtosis_historical` (Â±30%)
- Stocks typically have kurtosis 8-20
- Too low (<3): Missing extreme events
- Too high (>50): Generating too many extreme events

**5. ACF Lag-1 (Volatility Clustering)**
- Target: `acf_lag1_synthetic â‰ˆ acf_lag1_historical` (Â±20%)
- Positive ACF indicates volatility clustering (normal for finance)

### Stationarity Check

**Critical**: Always verify alpha + beta < 1

```python
alpha = result['garch_params']['alpha']
beta = result['garch_params']['beta']
persistence = alpha + beta

if persistence >= 1.0:
    print("âš ï¸ WARNING: Non-stationary model (infinite variance)")
else:
    print(f"âœ… Stationary: alpha + beta = {persistence:.3f}")
```


## Rate Limits

**Current Limits (Development):**
- No rate limiting enforced
- Maximum 10,000 scenarios per request
- Maximum 1,260 days horizon (5 years)

**Production Recommendations:**
- Implement rate limiting: 10 requests/minute per IP
- Cache fitted models (24-hour TTL)
- Queue long-running jobs (>1000 scenarios)

---

## Support & Feedback

**Logs:** Check service logs for detailed error messages
```bash
docker-compose logs -f garch
```

**Common Issues:**
1. **"Invalid ticker"**: Verify symbol exists on Yahoo Finance
2. **"Model did not converge"**: Increase data history or check for outliers
3. **"Non-stationary"**: Î± + Î² â‰¥ 1, model rejected

**Version:** 1.0  
**Last Updated:** January 14, 2026