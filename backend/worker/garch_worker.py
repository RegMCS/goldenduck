import os
import time
import logging
import sys
import importlib.util
from pathlib import Path

import yfinance as yf
import numpy as np
import pandas as pd
from scipy import stats

import io
import boto3
from goldenduck_core.redis_client import redis_client
from goldenduck_core.services.job_store import job_store
from goldenduck_core.models.enums import JobStatus
from worker.GARCH.services.garch_service import GARCHService
from worker.GARCH.services.scenarios import (
    generate_scenario,
    get_flash_crash_drift_schedule,
    get_flash_crash_theta_schedule,
    get_scenario_knobs,
)
from goldenduck_core.db.session import SessionLocal
from goldenduck_core.services.job_service import update_job_status

# Add ml_training to path for importing ParameterPredictor
ML_TRAINING_PATH = Path(__file__).parent.parent / "ml_training"
sys.path.insert(0, str(ML_TRAINING_PATH))

PREDICTOR_PATH = ML_TRAINING_PATH / "scripts" / "06_predict_parameters.py"
spec = importlib.util.spec_from_file_location(
    "predict_parameters_module", PREDICTOR_PATH
)
predict_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(predict_module)
predict_parameters = predict_module.predict_parameters

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("garch-worker")
MIN_CSV_DATA_POINTS = 500

# Cache the risk-free rate so we only fetch it once per worker process
_cached_rf_rate: float | None = None

FLASH_CRASH_HORIZON = 500


def _infer_skew_shock_flags(returns: np.ndarray) -> tuple[bool, dict]:
    """
    Infer whether to enable skew-aware shocks based on historical return skewness.

    Rule:
    - Enable skew-aware shocks if sample skew is meaningfully negative.
    - Use both magnitude and sampling-noise threshold for robustness.
    """
    clean = np.asarray(returns).flatten()
    clean = clean[np.isfinite(clean)]
    n = int(len(clean))

    if n < 50:
        return False, {
            "historical_skewness": 0.0,
            "skew_threshold": 0.0,
            "sample_size": n,
            "reason": "insufficient_samples",
        }

    sample_skew = float(stats.skew(clean, bias=False))
    # Approximate std error of sample skewness under normality.
    skew_se = float(np.sqrt(6.0 / n))
    threshold = max(0.10, skew_se)
    enable = bool(sample_skew < -threshold)

    return enable, {
        "historical_skewness": sample_skew,
        "skew_threshold": threshold,
        "sample_size": n,
        "reason": "negative_skew_detected" if enable else "not_negative_enough",
    }


def _get_risk_free_rate() -> float:
    """
    Fetch the annualised risk-free rate from the 13-week US T-bill yield (^IRX).
    Falls back to 4% if the fetch fails. Result is cached for the process lifetime.
    """
    global _cached_rf_rate
    if _cached_rf_rate is not None:
        return _cached_rf_rate
    try:
        tbill = yf.download("^IRX", period="5d", progress=False, threads=False)
        if not tbill.empty:
            close = tbill["Close"]
            # yfinance ≥0.2 returns a MultiIndex DataFrame for single tickers;
            # squeeze to a plain Series if needed.
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
            rf = float(close.iloc[-1]) / 100  # ^IRX is quoted in percent
            _cached_rf_rate = rf
            logger.info(f"Risk-free rate fetched from ^IRX: {rf:.4%}")
            return rf
    except Exception as e:
        logger.warning(
            f"Could not fetch risk-free rate from ^IRX: {e}. Using 4% fallback."
        )
    _cached_rf_rate = 0.04
    return _cached_rf_rate


def _series_stats(prices: list, returns_arr: np.ndarray) -> dict:
    from scipy import stats as scipy_stats

    n = len(returns_arr)
    if n == 0:
        return {
            "mean": 0.0,
            "std": 0.0,
            "skewness": 0.0,
            "kurtosis": 0.0,
            "var95": 0.0,
            "hurstMomentum": 0.5,
            "maxDrawdown": 0.0,
            "sharpe": 0.0,
            "annualizedReturn": 0.0,
            "annualizedVol": 0.0,
            "totalReturn": 0.0,
            "numDataPoints": 0,
        }

    mean_r = float(np.mean(returns_arr))
    std_r = float(np.std(returns_arr, ddof=1)) if n > 1 else 0.0
    ann_return = float((1 + mean_r) ** 252 - 1)
    ann_vol = float(std_r * np.sqrt(252))
    rf = _get_risk_free_rate()
    sharpe = (ann_return - rf) / ann_vol if ann_vol > 0 else 0.0

    peak = prices[0]
    max_dd = 0.0
    for p in prices:
        peak = max(peak, p)
        dd = (p - peak) / peak
        if dd < max_dd:
            max_dd = dd

    total_return = (prices[-1] - prices[0]) / prices[0] if prices[0] != 0 else 0.0

    # Daily VaR (95%) reported as positive loss magnitude.
    q05 = float(np.quantile(returns_arr, 0.05))
    var95 = max(0.0, -q05)

    # Hurst-based momentum score in [0, 1] (0.5 is near-random neutral).
    hurst_momentum = _safe_hurst_momentum(returns_arr)

    return {
        "mean": round(mean_r, 6),
        "std": round(std_r, 6),
        "skewness": round(float(scipy_stats.skew(returns_arr)), 4),
        "kurtosis": round(float(scipy_stats.kurtosis(returns_arr)), 4),
        "var95": round(var95, 6),
        "hurstMomentum": round(hurst_momentum, 4),
        "maxDrawdown": round(max_dd, 6),
        "sharpe": round(sharpe, 4),
        "annualizedReturn": round(ann_return, 6),
        "annualizedVol": round(ann_vol, 6),
        "totalReturn": round(total_return, 6),
        "numDataPoints": n,
    }


def compute_chart_data(historical_df: pd.DataFrame, scenario: pd.DataFrame, all_scenarios: list = None) -> dict:
    df = historical_df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index()
    date_col = "Date" if "Date" in df.columns else df.columns[0]

    historical = []
    for _, row in df.iterrows():
        historical.append(
            {
                "date": str(row[date_col])[:10],
                "open": round(float(row.get("Open", 0)), 4),
                "high": round(float(row.get("High", 0)), 4),
                "low": round(float(row.get("Low", 0)), 4),
                "close": round(float(row.get("Close", 0)), 4),
                "volume": int(row.get("Volume", 0)),
            }
        )

    last_date = pd.Timestamp(historical[-1]["date"])
    synth_dates = pd.bdate_range(last_date + pd.offsets.BDay(1), periods=len(scenario))
    synth_df = scenario.reset_index(drop=True)

    synthetic = []
    for i, date in enumerate(synth_dates):
        row = synth_df.iloc[i]
        synthetic.append(
            {
                "date": str(date)[:10],
                "open": round(float(row.get("Open", row.get("open", 0))), 4),
                "high": round(float(row.get("High", row.get("high", 0))), 4),
                "low": round(float(row.get("Low", row.get("low", 0))), 4),
                "close": round(float(row.get("Close", row.get("close", 0))), 4),
                "volume": int(row.get("Volume", row.get("volume", 0))),
            }
        )

    hist_closes = [h["close"] for h in historical]
    synth_closes = [s["close"] for s in synthetic]
    min_len = min(len(historical), len(synthetic))

    h_start = hist_closes[0] or 1.0
    s_start = synth_closes[0] or 1.0

    time_series = []
    for i in range(min_len):
        ts = int(pd.Timestamp(historical[i]["date"]).timestamp() * 1000)
        time_series.append(
            {
                "date": historical[i]["date"],
                "timestamp": ts,
                "historical": round(hist_closes[i] / h_start * 100, 4),
                "synthetic": round(synth_closes[i] / s_start * 100, 4),
            }
        )

    returns_data = []
    h_ret_list, s_ret_list = [], []
    h_cum, s_cum = 1.0, 1.0
    for i in range(1, min_len):
        h_ret = (
            (hist_closes[i] - hist_closes[i - 1]) / hist_closes[i - 1]
            if hist_closes[i - 1]
            else 0.0
        )
        s_ret = (
            (synth_closes[i] - synth_closes[i - 1]) / synth_closes[i - 1]
            if synth_closes[i - 1]
            else 0.0
        )
        h_cum *= 1 + h_ret
        s_cum *= 1 + s_ret
        h_ret_list.append(h_ret)
        s_ret_list.append(s_ret)
        ts = int(pd.Timestamp(historical[i]["date"]).timestamp() * 1000)
        returns_data.append(
            {
                "date": historical[i]["date"],
                "timestamp": ts,
                "historicalReturn": round(h_ret, 6),
                "syntheticReturn": round(s_ret, 6),
                "historicalCumReturn": round(h_cum - 1, 6),
                "syntheticCumReturn": round(s_cum - 1, 6),
            }
        )

    h_peak, s_peak = hist_closes[0], synth_closes[0]
    drawdowns = []
    for i in range(min_len):
        h_peak = max(h_peak, hist_closes[i])
        s_peak = max(s_peak, synth_closes[i])
        ts = int(pd.Timestamp(historical[i]["date"]).timestamp() * 1000)
        drawdowns.append(
            {
                "date": historical[i]["date"],
                "timestamp": ts,
                "historicalDrawdown": round((hist_closes[i] - h_peak) / h_peak, 6),
                "syntheticDrawdown": round((synth_closes[i] - s_peak) / s_peak, 6),
            }
        )

    stats = {
        "historical": _series_stats(hist_closes[:min_len], np.array(h_ret_list)),
        "synthetic": _series_stats(synth_closes[:min_len], np.array(s_ret_list)),
    }

    # Compute volatility fan chart (percentiles across all scenarios)
    volatility_fan = []
    if all_scenarios and len(all_scenarios) > 0:
        # Extract closes from all scenarios, normalize to index 100
        prices_matrix = []
        for s in all_scenarios:
            s_df = s.reset_index(drop=True)
            closes = [float(s_df.iloc[i].get("Close", s_df.iloc[i].get("close", 0))) for i in range(len(s_df))]
            if len(closes) > 0:
                start_price = closes[0] or 1.0
                normalized = [round(c / start_price * 100, 4) for c in closes]
                prices_matrix.append(normalized)

        if prices_matrix:
            prices_matrix = np.array(prices_matrix)
            # Compute percentiles at each timestep
            p10 = np.percentile(prices_matrix, 10, axis=0)
            p50 = np.percentile(prices_matrix, 50, axis=0)
            p90 = np.percentile(prices_matrix, 90, axis=0)

            # Align with historical dates (use same date range as synthetic)
            for i in range(min(len(synth_dates), len(p10), len(p50), len(p90))):
                date_str = str(synth_dates[i])[:10]
                ts = int(date_str.replace("-", "")) * 10000
                volatility_fan.append(
                    {
                        "date": date_str,
                        "timestamp": ts,
                        "historical": round(hist_closes[i] / h_start * 100, 4) if i < len(hist_closes) else None,
                        "p10": round(float(p10[i]), 4),
                        "p50": round(float(p50[i]), 4),
                        "p90": round(float(p90[i]), 4),
                    }
                )

    return {
        "historical": historical,
        "synthetic": synthetic,
        "timeSeries": time_series,
        "returns": returns_data,
        "drawdowns": drawdowns,
        "stats": stats,
        "volatilityFan": volatility_fan,
    }


def _safe_hurst_momentum(returns_arr: np.ndarray) -> float:
    """
    Estimate Hurst exponent from returns and map to momentum score in [0, 1].
    Neutral/noise-like behavior is around 0.5.
    """
    x = np.asarray(returns_arr).ravel()
    x = x[np.isfinite(x)]
    if len(x) < 20:
        return 0.5

    # Use cumulative demeaned series for robust scaling estimate.
    y = np.cumsum(x - float(np.mean(x)))
    if float(np.std(y)) < 1e-12:
        return 0.5

    lags = np.array([2, 4, 8, 16, 32, 64], dtype=int)
    lags = lags[lags < (len(y) // 2)]
    if len(lags) < 2:
        return 0.5

    log_lags = []
    log_tau = []
    for lag in lags:
        diff = y[lag:] - y[:-lag]
        tau = float(np.std(diff))
        if tau > 1e-12 and np.isfinite(tau):
            log_lags.append(np.log(float(lag)))
            log_tau.append(np.log(tau))

    if len(log_lags) < 2:
        return 0.5

    try:
        hurst = float(np.polyfit(np.array(log_lags), np.array(log_tau), 1)[0])
    except Exception:
        return 0.5

    if not np.isfinite(hurst):
        return 0.5
    return float(np.clip(hurst, 0.0, 1.0))


def _target_hurst_from_momentum_knob(
    desired_momentum: float,
    input_hurst: float,
    low_anchor: float = 0.15,
    high_anchor: float = 0.85,
) -> float:
    """
    Map momentum knob to target H with anchors:
      - knob=0.5 preserves input_hurst
      - knob=0.0 targets low_anchor  (~0.1-0.2)
      - knob=1.0 targets high_anchor (~0.8-0.9)
    """
    m = float(np.clip(desired_momentum, 0.0, 1.0))
    h_in = float(np.clip(input_hurst, 0.0, 1.0))
    lo = float(np.clip(low_anchor, 0.0, 1.0))
    hi = float(np.clip(high_anchor, 0.0, 1.0))

    if m <= 0.5:
        # Linear from (0, lo) to (0.5, h_in)
        t = m / 0.5 if 0.5 > 0 else 0.0
        target = lo + t * (h_in - lo)
    else:
        # Linear from (0.5, h_in) to (1, hi)
        t = (m - 0.5) / 0.5
        target = h_in + t * (hi - h_in)

    return float(np.clip(target, 0.0, 1.0))


def _get_log_returns_from_scenario(scenario: pd.DataFrame) -> np.ndarray:
    """
    Extract daily log returns from scenario close prices.
    """
    close = scenario["Close"].values.astype(float)
    if len(close) < 2:
        return np.array([], dtype=float)

    # Clip for numerical safety
    close = np.clip(close, 1e-12, None)
    return np.log(close[1:] / close[:-1])


def _count_extreme_events(returns: np.ndarray, threshold: float = 2.5) -> int:
    """
    Count number of returns exceeding threshold in standard deviations.
    Used for fat_tails knob selection.
    """
    if len(returns) < 2:
        return 0
    
    std = float(np.std(returns))
    if std < 1e-12:
        return 0
    
    normalized = np.abs(returns) / std
    count = int(np.sum(normalized > threshold))
    return count


def _safe_excess_kurtosis(returns: np.ndarray) -> float:
    """Robust finite excess kurtosis estimate for return series."""
    x = np.asarray(returns, dtype=float).ravel()
    x = x[np.isfinite(x)]
    if len(x) < 4:
        return 0.0
    try:
        k = float(stats.kurtosis(x))
    except Exception:
        return 0.0
    if not np.isfinite(k):
        return 0.0
    return k


def _baseline_match_distance(
    scenario: pd.DataFrame,
    historical_returns: np.ndarray,
) -> float:
    """
    Composite distance between one synthetic path and input returns.
    Lower is better.
    """
    h = np.asarray(historical_returns).ravel()
    h = h[np.isfinite(h)]
    s = _get_log_returns_from_scenario(scenario)
    s = s[np.isfinite(s)]

    if len(h) < 2 or len(s) < 2:
        return float("inf")

    # Input stats
    h_mean = float(np.mean(h))
    h_std = float(np.std(h, ddof=1)) if len(h) > 1 else 0.0
    h_ann_vol = float(h_std * np.sqrt(252.0))
    h_hurst = _safe_hurst_momentum(h)
    h_kurt = float(stats.kurtosis(h))
    h_skew = float(stats.skew(h))

    # Scenario stats
    s_mean = float(np.mean(s))
    s_std = float(np.std(s, ddof=1)) if len(s) > 1 else 0.0
    s_ann_vol = float(s_std * np.sqrt(252.0))
    s_hurst = _safe_hurst_momentum(s)
    s_kurt = float(stats.kurtosis(s))
    s_skew = float(stats.skew(s))

    if not all(
        np.isfinite(v)
        for v in [
            h_mean,
            h_ann_vol,
            h_hurst,
            h_kurt,
            h_skew,
            s_mean,
            s_ann_vol,
            s_hurst,
            s_kurt,
            s_skew,
        ]
    ):
        return float("inf")

    # Weighted normalized Euclidean distance.
    # Prioritize realized volatility and Hurst for visual/path-shape fidelity.
    w_mean = 0.20
    w_vol = 0.35
    w_hurst = 0.25
    w_kurt = 0.15
    w_skew = 0.05

    d_mean = (s_mean - h_mean) / max(abs(h_mean), 1e-4)
    d_vol = (s_ann_vol - h_ann_vol) / max(abs(h_ann_vol), 1e-6)
    d_hurst = (s_hurst - h_hurst)  # already naturally scaled in [0, 1]
    d_kurt = (s_kurt - h_kurt) / max(abs(h_kurt), 1.0)
    d_skew = (s_skew - h_skew) / max(abs(h_skew), 0.25)

    return float(
        np.sqrt(
            w_mean * d_mean * d_mean
            + w_vol * d_vol * d_vol
            + w_hurst * d_hurst * d_hurst
            + w_kurt * d_kurt * d_kurt
            + w_skew * d_skew * d_skew
        )
    )


def _path_tortuosity_ratio(scenario: pd.DataFrame) -> float:
    """
    Volatility path roughness heuristic:
      ratio = total path length / net displacement

    Low-vol path: ratio ~ 1 (straighter)
    High-vol path: ratio >> 1 (more winding)
    """
    close = scenario["Close"].values.astype(float)
    if len(close) < 2:
        return 1.0

    diffs = np.diff(close)
    path_length = float(np.sum(np.abs(diffs)))
    net_displacement = float(abs(close[-1] - close[0]))

    # Avoid division blow-ups when start/end are very close.
    denom = max(net_displacement, 1e-8)
    ratio = path_length / denom
    if not np.isfinite(ratio):
        return 1e6
    return float(ratio)


def _is_flash_crash_preset(user_knobs: dict, eps: float = 1e-9) -> bool:
    """Check whether current knobs match flash-crash preset values."""
    preset = get_scenario_knobs("flash_crash")
    return (
        abs(float(user_knobs.get("desired_trend", 0.0)) - float(preset.get("desired_trend", 0.0))) <= eps
        and abs(float(user_knobs.get("desired_volatility", 1.0)) - float(preset.get("desired_volatility", 1.0))) <= eps
        and abs(float(user_knobs.get("desired_fat_tails", 1.0)) - float(preset.get("desired_fat_tails", 1.0))) <= eps
        and abs(float(user_knobs.get("desired_momentum", 0.5)) - float(preset.get("desired_momentum", 0.5))) <= eps
    )


def _is_bull_run_preset(user_knobs: dict, eps: float = 5e-3) -> bool:
    """Check whether current knobs match bull-run preset values with a practical tolerance."""
    return (
        abs(float(user_knobs.get("desired_volatility", 1.0)) - 0.5) <= eps
        and abs(float(user_knobs.get("desired_fat_tails", 1.0)) - 0.6) <= eps
        and abs(float(user_knobs.get("desired_trend", 0.0)) - 0.7) <= eps
        and abs(float(user_knobs.get("desired_momentum", 0.5)) - 0.85) <= eps
    )


def _target_mean_from_desired_trend(desired_trend: float) -> float:
    """
    Map desired trend knob to target DAILY mean log return.
    Piecewise annual drift mapping:
      0.25 -> 10%/yr, 0.50 -> 25%/yr, 1.00 -> 50%/yr (signed).
    """
    trend_mag = float(np.clip(abs(desired_trend), 0.0, 1.0))
    if trend_mag <= 0.25:
        annual_drift_mag = 0.40 * trend_mag
    elif trend_mag <= 0.50:
        annual_drift_mag = 0.10 + 0.60 * (trend_mag - 0.25)
    else:
        annual_drift_mag = 0.25 + 0.50 * (trend_mag - 0.50)

    annual_drift = np.sign(desired_trend) * annual_drift_mag
    return float(annual_drift / 252.0)


def _score_bull_run_path(
    scenario: pd.DataFrame,
    target_mean: float,
    target_vol: float,
    max_dd_threshold: float = 0.20,
) -> float:
    """
    Composite score for bull-run preset path selection.
    Hard disqualifiers:
      - Negative net return
      - Max drawdown > max_dd_threshold
    """
    close = scenario["Close"].values.astype(float)
    close = np.clip(close, 1e-12, None)
    if len(close) < 2:
        return float("-inf")

    r = np.log(close[1:] / close[:-1])

    net_return = (close[-1] - close[0]) / close[0]
    if net_return < 0:
        return float("-inf")

    peak = np.maximum.accumulate(close)
    drawdown = (close - peak) / peak
    max_dd = abs(float(np.min(drawdown)))
    if max_dd > max_dd_threshold:
        return float("-inf")

    mean_r = float(np.mean(r)) if len(r) > 0 else 0.0
    std_r = float(np.std(r, ddof=1)) if len(r) > 1 else 0.0

    trend_score = 1.0 - abs(mean_r - target_mean) / (abs(target_mean) + 1e-6)
    vol_score = 1.0 - abs(std_r - target_vol) / (abs(target_vol) + 1e-6)
    dd_score = 1.0 - (max_dd / max_dd_threshold)

    # Clamp to keep each component in [0, 1] range when possible.
    trend_score = float(np.clip(trend_score, 0.0, 1.0))
    vol_score = float(np.clip(vol_score, 0.0, 1.0))
    dd_score = float(np.clip(dd_score, 0.0, 1.0))

    return float(0.5 * trend_score + 0.3 * dd_score + 0.2 * vol_score)


def _is_valid_flash_crash(
    prices: np.ndarray, crash_start: int, crash_days: int = 5
) -> bool:
    """
    Hard filters for flash-crash shape:
      - crash depth >= 15%
      - recovery >= 50%
      - end price >= start price (no full-period decline)
    """
    p = np.asarray(prices, dtype=float)
    if len(p) < 3:
        return False

    crash_start = int(np.clip(crash_start, 1, max(len(p) - 2, 1)))
    crash_end = min(crash_start + max(int(crash_days), 1), len(p))
    if crash_end <= crash_start:
        return False

    pre_crash = float(p[crash_start - 1])
    crash_low = float(np.min(p[crash_start:crash_end]))
    end_price = float(p[-1])

    if pre_crash <= 1e-12:
        return False

    crash_depth = (pre_crash - crash_low) / pre_crash
    denom = (pre_crash - crash_low)
    if denom <= 1e-12:
        return False
    recovery = (end_price - crash_low) / denom

    if crash_depth < 0.15:
        return False
    if recovery < 0.50:
        return False
    if end_price < float(p[0]):
        return False
    return True


def _score_flash_crash_path(
    prices: np.ndarray, crash_start: int, crash_days: int = 5
) -> float:
    """
    Composite score for flash-crash path quality.
    Assumes hard filters are already satisfied.
    """
    p = np.asarray(prices, dtype=float)
    if len(p) < 3:
        return float("-inf")

    crash_start = int(np.clip(crash_start, 1, max(len(p) - 2, 1)))
    crash_end = min(crash_start + max(int(crash_days), 1), len(p))
    if crash_end <= crash_start:
        return float("-inf")

    pre_crash_prices = p[:crash_start]
    if len(pre_crash_prices) < 2:
        return float("-inf")

    local_window = p[crash_start:crash_end]
    if len(local_window) == 0:
        return float("-inf")

    crash_low_local_idx = int(np.argmin(local_window))
    crash_low_idx = crash_start + crash_low_local_idx
    crash_low = float(p[crash_low_idx])
    pre_crash = float(p[crash_start - 1])
    end_price = float(p[-1])

    pre_crash_returns = np.diff(np.log(np.clip(pre_crash_prices, 1e-12, None)))
    calm_score = 1.0 - min(float(np.std(pre_crash_returns)) / 0.015, 1.0)

    crash_depth = (pre_crash - crash_low) / max(pre_crash, 1e-12)
    crash_score = min(crash_depth / 0.30, 1.0)

    denom = max(pre_crash - crash_low, 1e-12)
    recovery = (end_price - crash_low) / denom
    recovery_score = min(recovery, 1.0)

    post_crash = p[crash_low_idx:]
    if len(post_crash) >= 2:
        post_returns = np.diff(np.log(np.clip(post_crash, 1e-12, None)))
        vshape_score = 1.0 if float(np.mean(post_returns)) > 0 else 0.0
    else:
        vshape_score = 0.0

    return float(
        0.25 * calm_score
        + 0.35 * crash_score
        + 0.25 * recovery_score
        + 0.15 * vshape_score
    )


def _score_flash_crash_path_with_breakdown(
    prices: np.ndarray,
    returns: np.ndarray,
    sigma_path: np.ndarray,
    historical_prices: np.ndarray,
    delta_schedule: np.ndarray,
    trigger_start: int,
    trigger_end: int,
    recovery_end: int,
    baseline_sigma: float,
) -> tuple[float, list[dict]]:
    """Weighted flash-crash evaluation with per-criterion breakdown."""
    p = np.asarray(prices, dtype=float)
    r = np.asarray(returns, dtype=float)
    s = np.asarray(sigma_path, dtype=float)
    h = np.asarray(historical_prices, dtype=float)

    p = p[np.isfinite(p)]
    r = r[np.isfinite(r)]
    s = s[np.isfinite(s)]
    h = h[np.isfinite(h)]
    d = np.asarray(delta_schedule, dtype=float)
    d = d[np.isfinite(d)]

    if len(p) < 3:
        return 0.0, []

    trigger_start = int(np.clip(trigger_start, 1, len(p) - 2))
    trigger_end = int(np.clip(trigger_end, trigger_start + 1, len(p) - 1))
    recovery_end = int(np.clip(recovery_end, trigger_end + 1, len(p) - 1))
    crash_bottom_idx = trigger_start + int(np.argmin(p[trigger_start:trigger_end]))

    pre_crash_peak = float(np.max(p[:trigger_start])) if trigger_start > 0 else float(p[0])
    crash_trough = float(p[crash_bottom_idx])
    drawdown = (pre_crash_peak - crash_trough) / max(pre_crash_peak, 1e-12)
    crash_depth_target = 0.25
    crash_depth_score = float(
        np.clip(1.0 - abs(drawdown - crash_depth_target) / crash_depth_target, 0.0, 1.0)
    )

    min_len = min(len(s), len(d))
    if min_len >= 2:
        sigma_slice = s[:min_len]
        delta_slice = d[:min_len]
        if np.std(sigma_slice) > 1e-12 and np.std(delta_slice) > 1e-12:
            corr = float(np.corrcoef(sigma_slice, delta_slice)[0, 1])
        else:
            corr = 0.0
    else:
        corr = 0.0
    vol_alignment_score = float(np.clip(max(0.0, corr), 0.0, 1.0))

    crash_window = p[max(0, trigger_end - 10) : min(len(p), trigger_end + 5)]
    trough = float(np.min(crash_window)) if len(crash_window) > 0 else crash_trough
    recovered = float(p[recovery_end])
    pre_crash_ref = float(np.max(p[: max(1, trigger_end - 20)]))
    recovery_ratio = (recovered - trough) / max(pre_crash_ref - trough, 1e-8)
    recovery_target = 0.55
    recovery_score = float(
        np.clip(1.0 - abs(recovery_ratio - recovery_target) / recovery_target, 0.0, 1.0)
    )

    crash_returns = r[trigger_start : min(recovery_end, len(r))]
    if len(crash_returns) >= 3:
        skew_val = float(stats.skew(crash_returns, bias=False))
        skewness_score = 0.0 if skew_val >= 0 else float(min(1.0, abs(skew_val) / 1.5))
    else:
        skew_val = 0.0
        skewness_score = 0.0

    baseline_sigma = max(float(baseline_sigma), 1e-8)
    pre_vol = float(np.mean(s[:trigger_start])) if len(s[:trigger_start]) > 0 else baseline_sigma
    pre_vol_ratio = pre_vol / baseline_sigma
    pre_calm_score = float(np.clip(1.0 - abs(pre_vol_ratio - 1.0), 0.0, 1.0))

    criteria = [
        {
            "key": "crash_depth",
            "label": "Crash Depth",
            "weight": 0.30,
            "score": crash_depth_score,
            "target": crash_depth_target,
            "actual": float(drawdown),
        },
        {
            "key": "vol_alignment",
            "label": "Vol Alignment",
            "weight": 0.25,
            "score": vol_alignment_score,
            "target": 1.0,
            "actual": float(corr),
        },
        {
            "key": "recovery",
            "label": "Recovery Shape",
            "weight": 0.20,
            "score": recovery_score,
            "target": recovery_target,
            "actual": float(recovery_ratio),
        },
        {
            "key": "skewness",
            "label": "Crash Skewness",
            "weight": 0.15,
            "score": skewness_score,
            "target": -0.5,
            "actual": float(skew_val),
        },
        {
            "key": "pre_calm",
            "label": "Pre-crash Calm",
            "weight": 0.10,
            "score": pre_calm_score,
            "target": 1.0,
            "actual": float(pre_vol_ratio),
        },
    ]

    total = float(sum(item["weight"] * item["score"] for item in criteria))
    return total, criteria


def _select_best_display_scenario(
    scenarios: list[pd.DataFrame],
    user_knobs: dict,
    historical_prices: np.ndarray | None = None,
    historical_returns: np.ndarray | None = None,
    scenario_metadata: list[dict] | None = None,
    delta_schedule: np.ndarray | None = None,
) -> tuple[int, str, float, float, float, dict]:
    """
    Select one scenario path for frontend display using a single-knob visual objective.

    Rules (single-knob assumption):
      - Trend knob active: choose path with mean log return closest to target mean.
            - Momentum knob active: choose path with Hurst momentum closest to target [0, 1].
      - Fat Tails knob active: choose path with extreme event count closest to target.
      - Otherwise: fallback to first path.

    Returns:
            (best_index, objective_name, target_value, best_value, best_score, breakdown)
    """
    if not scenarios:
                return 0, "fallback", 0.0, 0.0, 0.0, {"total": 0.0, "criteria": []}

    desired_volatility = float(user_knobs.get("desired_volatility", 1.0))
    desired_trend = float(user_knobs.get("desired_trend", 0.0))
    desired_fat_tails = float(user_knobs.get("desired_fat_tails", 1.0))
    desired_momentum = float(user_knobs.get("desired_momentum", 0.5))

    # Detect "single knob changed" mode (with tiny tolerance)
    eps = 1e-12
    trend_active = abs(desired_trend - 0.0) > eps
    momentum_active = abs(desired_momentum - 0.5) > eps
    volatility_active = abs(desired_volatility - 1.0) > eps
    fat_tails_active = abs(desired_fat_tails - 1.0) > eps

    # Bull-run preset: volatility=0.5, fat_tails=0.6, trend=0.7, momentum=0.85
    bull_run_preset = _is_bull_run_preset(user_knobs)

    if bull_run_preset:
        target_mean = _target_mean_from_desired_trend(desired_trend)
        hist = np.asarray(historical_returns).ravel() if historical_returns is not None else np.array([])
        hist = hist[np.isfinite(hist)] if len(hist) > 0 else hist
        hist_std = float(np.std(hist, ddof=1)) if len(hist) > 1 else 0.01
        target_vol = max(1e-8, hist_std * desired_volatility)

        best_idx = 0
        best_score = float("-inf")
        for i, scenario in enumerate(scenarios):
            score = _score_bull_run_path(
                scenario=scenario,
                target_mean=target_mean,
                target_vol=target_vol,
                max_dd_threshold=0.20,
            )
            if score > best_score:
                best_score = score
                best_idx = i

        if not np.isfinite(best_score):
            # No path passed hard filters; keep objective tag but return finite score.
            return 0, "bull_run_composite", 1.0, 0.0, 0.0, {"total": 0.0, "criteria": []}
        return best_idx, "bull_run_composite", 1.0, float(best_score), float(best_score), {
            "total": float(best_score),
            "criteria": [],
        }

    if _is_flash_crash_preset(user_knobs):
        horizon = len(scenarios[0]) if scenarios else 0
        trigger_start = int(0.60 * horizon)
        trigger_end = int(0.70 * horizon)
        recovery_end = int(0.90 * horizon)

        hist_prices = (
            np.asarray(historical_prices, dtype=float).ravel()
            if historical_prices is not None
            else np.array([])
        )
        hist_prices = hist_prices[np.isfinite(hist_prices)] if len(hist_prices) > 0 else hist_prices

        hist = np.asarray(historical_returns).ravel() if historical_returns is not None else np.array([])
        hist = hist[np.isfinite(hist)] if len(hist) > 0 else hist
        baseline_sigma = float(np.std(hist, ddof=1)) if len(hist) > 1 else 0.01

        best_idx = 0
        best_score = float("-inf")
        best_breakdown: list[dict] = []
        for i, scenario in enumerate(scenarios):
            prices = scenario["Close"].values.astype(float)
            prices = np.clip(prices, 1e-12, None)
            returns = np.log(prices[1:] / prices[:-1])
            meta = scenario_metadata[i] if scenario_metadata and i < len(scenario_metadata) else {}
            sigma_path = np.asarray(meta.get("volatility_forecast", []), dtype=float)
            if len(sigma_path) == 0:
                sigma_path = np.full(len(prices), baseline_sigma, dtype=float)

            score, breakdown = _score_flash_crash_path_with_breakdown(
                prices=prices,
                returns=returns,
                sigma_path=sigma_path,
                historical_prices=hist_prices,
                delta_schedule=np.asarray(delta_schedule if delta_schedule is not None else np.array([]), dtype=float),
                trigger_start=trigger_start,
                trigger_end=trigger_end,
                recovery_end=recovery_end,
                baseline_sigma=baseline_sigma,
            )

            if score > best_score:
                best_score = score
                best_idx = i
                best_breakdown = breakdown

        if not np.isfinite(best_score):
            # No path passed hard filters; keep objective tag but return finite score.
            return 0, "flash_crash_evaluation", 1.0, 0.0, 0.0, {"total": 0.0, "criteria": []}
        return best_idx, "flash_crash_evaluation", 1.0, float(best_score), float(best_score), {
            "total": float(best_score),
            "criteria": best_breakdown,
        }

    # Weighted primary + preservation selection against input CSV stats.
    # If one knob is tweaked: 0.55 weight for active knob target, 0.15 each for
    # preserving untouched knobs close to historical input stats.
    # If no knobs are tweaked: equal 0.25 weight across all 4 knobs to preserve input.
    if historical_returns is None:
        return 0, "fallback", 0.0, 0.0, 0.0, {"total": 0.0, "criteria": []}

    hist = np.asarray(historical_returns, dtype=float).ravel()
    hist = hist[np.isfinite(hist)]
    if len(hist) < 2:
        return 0, "fallback", 0.0, 0.0, 0.0, {"total": 0.0, "criteria": []}

    # Historical/base (input CSV) stats used for preservation terms.
    hist_mean = float(np.mean(hist))
    hist_std = float(np.std(hist, ddof=1)) if len(hist) > 1 else 0.0
    hist_std = max(hist_std, 1e-6)
    hist_hurst = _safe_hurst_momentum(hist)
    hist_kurt = _safe_excess_kurtosis(hist)

    # Target values for active knob objectives.
    target_std = float(hist_std * desired_volatility)
    target_mean = _target_mean_from_desired_trend(desired_trend)
    target_momentum = _target_hurst_from_momentum_knob(
        desired_momentum,
        hist_hurst,
        low_anchor=0.15,
        high_anchor=0.85,
    )

    # Multiplicative fat-tails target anchored to input CSV kurtosis.
    # multiplier = 1.0 at fat_tails=1.0, 1.15 at fat_tails=2.0, 0.925 at fat_tails=0.5.
    # The target kurtosis is scaled relative to the historical input kurtosis.
    fat = float(np.clip(desired_fat_tails, 0.5, 2.0))
    kurt_multiplier = float(1.0 + 0.15 * (fat - 1.0))
    target_kurt = float(np.clip(hist_kurt * kurt_multiplier, -1.0, 12.0))

    no_knob_active = (
        not trend_active
        and not momentum_active
        and not volatility_active
        and not fat_tails_active
    )
    vol_only = volatility_active and not trend_active and not momentum_active and not fat_tails_active
    trend_only = trend_active and not momentum_active and not volatility_active and not fat_tails_active
    momentum_only = momentum_active and not trend_active and not volatility_active and not fat_tails_active
    fat_only = fat_tails_active and not trend_active and not momentum_active and not volatility_active

    # Keep current behavior for unexpected multi-knob states.
    if not (no_knob_active or vol_only or trend_only or momentum_only or fat_only):
        return 0, "fallback", 0.0, 0.0, 0.0, {"total": 0.0, "criteria": []}

    def _norm_err(actual: float, target: float, denom: float) -> float:
        d = abs(float(actual) - float(target)) / max(float(denom), 1e-12)
        return float(np.clip(d, 0.0, 1.0))

    best_idx = 0
    best_score = float("inf")
    best_obj_val = 0.0
    best_breakdown: list[dict] = []

    # Objective metadata for frontend display.
    if vol_only:
        objective_name = "volatility_std"
        objective_target = float(target_std)
    elif trend_only:
        objective_name = "trend_mean"
        objective_target = float(target_mean)
    elif momentum_only:
        objective_name = "momentum_hurst"
        objective_target = float(target_momentum)
    elif fat_only:
        objective_name = "fat_tails_kurtosis"
        objective_target = float(target_kurt)
    else:
        objective_name = "baseline_composite_match"
        objective_target = 0.0

    trend_direction_target = float(target_mean) if trend_only else float(hist_mean)

    for i, scenario in enumerate(scenarios):
        r = _get_log_returns_from_scenario(scenario)
        r = r[np.isfinite(r)]
        if len(r) < 2:
            continue

        s_mean = float(np.mean(r))
        s_std = float(np.std(r, ddof=1)) if len(r) > 1 else 0.0
        s_hurst = _safe_hurst_momentum(r)
        s_kurt = _safe_excess_kurtosis(r)

        if not all(np.isfinite(v) for v in [s_mean, s_std, s_hurst, s_kurt]):
            continue

        # Hard disqualifier: reject scenarios whose trend sign flips relative to the active trend target.
        if (trend_direction_target > 0.0 and s_mean < 0.0) or (trend_direction_target < 0.0 and s_mean > 0.0):
            continue

        # Preservation terms (stay close to input CSV stats).
        d_vol_preserve = _norm_err(s_std, hist_std, hist_std)
        d_trend_preserve = _norm_err(s_mean, hist_mean, 0.5 * hist_std)
        d_momentum_preserve = _norm_err(s_hurst, hist_hurst, 0.5)
        d_fat_preserve = _norm_err(s_kurt, hist_kurt, max(abs(hist_kurt), 1.0))

        # Active-objective terms.
        d_vol_target = _norm_err(s_std, target_std, hist_std)
        d_trend_target = _norm_err(s_mean, target_mean, 0.5 * hist_std)
        d_momentum_target = _norm_err(s_hurst, target_momentum, 0.5)
        d_fat_target = _norm_err(s_kurt, target_kurt, max(abs(target_kurt), 1.0))

        if no_knob_active:
            score = (
                0.25 * d_vol_preserve
                + 0.25 * d_trend_preserve
                + 0.25 * d_momentum_preserve
                + 0.25 * d_fat_preserve
            )
            obj_val = float(score)
            scenario_breakdown = [
                {
                    "key": "vol_preserve",
                    "label": "Volatility Match",
                    "weight": 0.25,
                    "score": float(d_vol_preserve),
                    "target": float(hist_std),
                    "actual": float(s_std),
                },
                {
                    "key": "trend_preserve",
                    "label": "Trend Match",
                    "weight": 0.25,
                    "score": float(d_trend_preserve),
                    "target": float(hist_mean),
                    "actual": float(s_mean),
                },
                {
                    "key": "momentum_preserve",
                    "label": "Momentum Match",
                    "weight": 0.25,
                    "score": float(d_momentum_preserve),
                    "target": float(hist_hurst),
                    "actual": float(s_hurst),
                },
                {
                    "key": "fat_preserve",
                    "label": "Fat Tails Match",
                    "weight": 0.25,
                    "score": float(d_fat_preserve),
                    "target": float(hist_kurt),
                    "actual": float(s_kurt),
                },
            ]
        elif vol_only:
            score = (
                0.55 * d_vol_target
                + 0.15 * d_trend_preserve
                + 0.15 * d_momentum_preserve
                + 0.15 * d_fat_preserve
            )
            obj_val = float(s_std)
            scenario_breakdown = [
                {
                    "key": "vol_target",
                    "label": "Volatility Target",
                    "weight": 0.55,
                    "score": float(d_vol_target),
                    "target": float(target_std),
                    "actual": float(s_std),
                },
                {
                    "key": "trend_preserve",
                    "label": "Trend Match",
                    "weight": 0.15,
                    "score": float(d_trend_preserve),
                    "target": float(hist_mean),
                    "actual": float(s_mean),
                },
                {
                    "key": "momentum_preserve",
                    "label": "Momentum Match",
                    "weight": 0.15,
                    "score": float(d_momentum_preserve),
                    "target": float(hist_hurst),
                    "actual": float(s_hurst),
                },
                {
                    "key": "fat_preserve",
                    "label": "Fat Tails Match",
                    "weight": 0.15,
                    "score": float(d_fat_preserve),
                    "target": float(hist_kurt),
                    "actual": float(s_kurt),
                },
            ]
        elif trend_only:
            score = (
                0.55 * d_trend_target
                + 0.15 * d_vol_preserve
                + 0.15 * d_momentum_preserve
                + 0.15 * d_fat_preserve
            )
            obj_val = float(s_mean)
            scenario_breakdown = [
                {
                    "key": "trend_target",
                    "label": "Trend Target",
                    "weight": 0.55,
                    "score": float(d_trend_target),
                    "target": float(target_mean),
                    "actual": float(s_mean),
                },
                {
                    "key": "vol_preserve",
                    "label": "Volatility Match",
                    "weight": 0.15,
                    "score": float(d_vol_preserve),
                    "target": float(hist_std),
                    "actual": float(s_std),
                },
                {
                    "key": "momentum_preserve",
                    "label": "Momentum Match",
                    "weight": 0.15,
                    "score": float(d_momentum_preserve),
                    "target": float(hist_hurst),
                    "actual": float(s_hurst),
                },
                {
                    "key": "fat_preserve",
                    "label": "Fat Tails Match",
                    "weight": 0.15,
                    "score": float(d_fat_preserve),
                    "target": float(hist_kurt),
                    "actual": float(s_kurt),
                },
            ]
        elif momentum_only:
            score = (
                0.55 * d_momentum_target
                + 0.15 * d_vol_preserve
                + 0.15 * d_trend_preserve
                + 0.15 * d_fat_preserve
            )
            obj_val = float(s_hurst)
            scenario_breakdown = [
                {
                    "key": "momentum_target",
                    "label": "Momentum Target",
                    "weight": 0.55,
                    "score": float(d_momentum_target),
                    "target": float(target_momentum),
                    "actual": float(s_hurst),
                },
                {
                    "key": "vol_preserve",
                    "label": "Volatility Match",
                    "weight": 0.15,
                    "score": float(d_vol_preserve),
                    "target": float(hist_std),
                    "actual": float(s_std),
                },
                {
                    "key": "trend_preserve",
                    "label": "Trend Match",
                    "weight": 0.15,
                    "score": float(d_trend_preserve),
                    "target": float(hist_mean),
                    "actual": float(s_mean),
                },
                {
                    "key": "fat_preserve",
                    "label": "Fat Tails Match",
                    "weight": 0.15,
                    "score": float(d_fat_preserve),
                    "target": float(hist_kurt),
                    "actual": float(s_kurt),
                },
            ]
        else:  # fat_only
            score = (
                0.55 * d_fat_target
                + 0.15 * d_vol_preserve
                + 0.15 * d_trend_preserve
                + 0.15 * d_momentum_preserve
            )
            obj_val = float(s_kurt)
            scenario_breakdown = [
                {
                    "key": "fat_target",
                    "label": "Fat Tails Target",
                    "weight": 0.55,
                    "score": float(d_fat_target),
                    "target": float(target_kurt),
                    "actual": float(s_kurt),
                },
                {
                    "key": "vol_preserve",
                    "label": "Volatility Match",
                    "weight": 0.15,
                    "score": float(d_vol_preserve),
                    "target": float(hist_std),
                    "actual": float(s_std),
                },
                {
                    "key": "trend_preserve",
                    "label": "Trend Match",
                    "weight": 0.15,
                    "score": float(d_trend_preserve),
                    "target": float(hist_mean),
                    "actual": float(s_mean),
                },
                {
                    "key": "momentum_preserve",
                    "label": "Momentum Match",
                    "weight": 0.15,
                    "score": float(d_momentum_preserve),
                    "target": float(hist_hurst),
                    "actual": float(s_hurst),
                },
            ]

        if score < best_score:
            best_score = float(score)
            best_idx = i
            best_obj_val = float(obj_val)
            best_breakdown = scenario_breakdown

    if not np.isfinite(best_score):
        return 0, objective_name, float(objective_target), 0.0, 0.0, {"total": 0.0, "criteria": []}

    return best_idx, objective_name, float(objective_target), float(best_obj_val), float(best_score), {
        "total": float(best_score),
        "criteria": best_breakdown,
    }

    return 0, "fallback", 0.0, 0.0, 0.0


S3_BUCKET_NAME = os.environ.get("S3_BUCKET_NAME", "goldenduck-results")
s3_client = boto3.client("s3")


logger.info("GARCH worker started, waiting for jobs...")

while True:
    job_id = None
    try:
        # wait for job
        _, job_id = redis_client.brpop("queue:garch")
        logger.info("Picked up job %s", job_id)

        job = job_store.get_job(job_id)
        if not job:
            raise ValueError("Job metadata not found")

        job_store.set_status(job_id, "running")

        try:
            db = SessionLocal()
            update_job_status(db, job_id, JobStatus.running)
        except Exception as db_exc:
            logger.error("Failed to update DB for running job %s: %s", job_id, db_exc)
        finally:
            db.close()

        params = job["parameters"]

        ticker = params.get("ticker")
        csv_data = params.get("csv_data")
        horizon = int(params.get("horizon", 252))

        # Use fixed defaults for GARCH fitting (not exposed to user)
        p = 1
        q = 1
        num_scenarios = 100
        volatility_multiplier = 1.0

        # Extract user knobs for ML parameter prediction.
        # Skew flags may be auto-overridden later after data is loaded.
        user_knobs = {
            "ticker": ticker,
            "desired_volatility": float(params.get("desired_volatility", 1.0)),
            "desired_trend": float(params.get("desired_trend", 0.0)),
            "desired_fat_tails": float(params.get("desired_fat_tails", 1.0)),
            "desired_momentum": float(params.get("desired_momentum", 0.5)),
            "asset_class": params.get("asset_class") or params.get("assetClass"),
            # A/B toggle for asymmetric innovation shocks when dist='skewt'
            "use_skew_shocks": bool(params.get("use_skew_shocks", False)),
            # A/B toggle to force using 'skewt' branch for return shocks
            "force_skewt_distribution": bool(
                params.get("force_skewt_distribution", False)
            ),
        }

        # Load data from CSV or Yahoo Finance
        if csv_data:
            logger.info(
                "Running GARCH with uploaded CSV (p=%s, q=%s, scenarios=%s, horizon=%s)",
                p,
                q,
                num_scenarios,
                horizon,
            )
            # Parse CSV data
            from io import StringIO

            csv_buffer = StringIO(csv_data)
            data = pd.read_csv(csv_buffer)

            # Validate required columns (case-insensitive)
            required_cols = ["Open", "High", "Low", "Close", "Volume"]

            # Normalize column names to title case
            data.columns = [col.strip().title() for col in data.columns]

            # Check for required columns
            missing_cols = [col for col in required_cols if col not in data.columns]
            if missing_cols:
                raise ValueError(
                    f"CSV missing required columns: {missing_cols}. Found columns: {list(data.columns)}"
                )

            # Keep only the required OHLCV columns
            data = data[required_cols]

            # Ensure sufficient data for stable fitting/validation
            if len(data) < MIN_CSV_DATA_POINTS:
                raise ValueError(
                    f"CSV must contain at least {MIN_CSV_DATA_POINTS} data rows; found {len(data)}"
                )

            # Create a date index if not present (for uploaded CSV without dates)
            # Use recent dates working backwards from today
            end_date = pd.Timestamp.today()
            date_range = pd.date_range(end=end_date, periods=len(data), freq="D")
            data.index = date_range

            logger.info(
                f"Loaded {len(data)} rows from uploaded CSV with synthetic date range"
            )
        elif ticker:
            logger.info(
                "Running GARCH for %s (p=%s, q=%s, scenarios=%s, horizon=%s)",
                ticker,
                p,
                q,
                num_scenarios,
                horizon,
            )
            data = yf.download(
                ticker,
                period="2y",
                progress=False,
                threads=False,
            )

            if data.empty:
                raise ValueError(f"No market data returned for ticker {ticker}")
        else:
            raise ValueError("Either ticker or csv_data must be provided")

        if _is_flash_crash_preset(user_knobs) and horizon != FLASH_CRASH_HORIZON:
            logger.info(
                "Flash crash preset forces horizon=%s; overriding requested horizon=%s",
                FLASH_CRASH_HORIZON,
                horizon,
            )
            horizon = FLASH_CRASH_HORIZON

        garch = GARCHService()

        fitted_params = garch.fit_with_retry(data, p=p, q=q)

        # Predict delta and theta using ML
        logger.info(f"Predicting GARCH-FX parameters from user knobs...")
        returns = np.log(data["Close"].values[1:] / data["Close"].values[:-1])

        # Momentum control anchor: preserve historical H when knob=0.5,
        # pull towards low/high H anchors when knob moves to 0/1.
        historical_hurst = _safe_hurst_momentum(returns)
        target_hurst = _target_hurst_from_momentum_knob(
            user_knobs.get("desired_momentum", 0.5),
            historical_hurst,
            low_anchor=0.15,
            high_anchor=0.85,
        )
        user_knobs["historical_hurst"] = float(historical_hurst)
        user_knobs["target_hurst"] = float(target_hurst)

        logger.info(
            "Momentum mapping: knob=%.3f, H_input=%.4f, H_target=%.4f",
            float(user_knobs.get("desired_momentum", 0.5)),
            float(historical_hurst),
            float(target_hurst),
        )

        # Auto mode for skew flags:
        # If user omits a flag, infer from data skewness.
        auto_enable_skew, skew_meta = _infer_skew_shock_flags(returns)
        use_skew_user_provided = "use_skew_shocks" in params
        force_skewt_user_provided = "force_skewt_distribution" in params

        if not use_skew_user_provided:
            user_knobs["use_skew_shocks"] = auto_enable_skew
        if not force_skewt_user_provided:
            user_knobs["force_skewt_distribution"] = auto_enable_skew

        logger.info(
            "Skew flag decision: use_skew_shocks=%s, force_skewt_distribution=%s "
            "(user_provided_use=%s, user_provided_force=%s, hist_skew=%.4f, threshold=%.4f, n=%s, reason=%s)",
            user_knobs["use_skew_shocks"],
            user_knobs["force_skewt_distribution"],
            use_skew_user_provided,
            force_skewt_user_provided,
            skew_meta["historical_skewness"],
            skew_meta["skew_threshold"],
            skew_meta["sample_size"],
            skew_meta["reason"],
        )

        pred_params = predict_parameters(
            historical_returns=returns, user_knobs=user_knobs
        )

        # Check if bull run preset is active
        desired_volatility = user_knobs.get("desired_volatility", 1.0)
        desired_fat_tails = user_knobs.get("desired_fat_tails", 1.0)
        bull_run_preset = _is_bull_run_preset(user_knobs)

        # Only bypass RF predictor if NOT using bull run preset
        if not bull_run_preset:
            # TEMPORARY: Bypass RF predictor and use desired_volatility directly as delta
            pred_params["delta"] = desired_volatility
            pred_params["delta_confidence"] = 1.0  # High confidence since we're using user input directly
            logger.info(f"  Delta (ML): {pred_params['delta']:.4f} [USING DESIRED_VOLATILITY DIRECTLY]")
        else:
            # For bull run preset, use RF predictor's delta prediction
            logger.info(f"  Delta (ML): {pred_params['delta']:.4f} [RF PREDICTOR - BULL RUN PRESET]")

        # Map desired_fat_tails to theta (forecast stochasticity)
        # Range: 0.5 -> 1e-5, 2.0 -> 1e-2
        fat_tails_clipped = float(np.clip(desired_fat_tails, 0.5, 2.0))

        # Linear interpolation from 1e-5 to 1e-2
        theta = 1e-5 + (fat_tails_clipped - 0.5) / 1.5 * (1e-2 - 1e-5)
        pred_params["theta"] = theta

        theta_sequence = None
        drift_sequence = None
        if _is_flash_crash_preset(user_knobs):
            theta_sequence = get_flash_crash_theta_schedule(horizon)
            drift_sequence = get_flash_crash_drift_schedule(horizon)
            logger.info("Using flash_crash theta scheduler")
            logger.info("Using flash_crash drift scheduler")

        logger.info(f"  Theta (mapped from fat_tails={desired_fat_tails}): {pred_params['theta']:.6f}")

        if _is_flash_crash_preset(user_knobs):
            delta_sequence, flash_desc = generate_scenario("flash_crash", horizon)
            logger.info("Using flash_crash preset delta sequence: %s", flash_desc)
        else:
            delta_sequence = np.full(horizon, float(pred_params["delta"]))

        scenarios = garch.generate_scenarios_fx(
            num_scenarios=num_scenarios,
            horizon=horizon,
            theta=float(pred_params["theta"]),
            theta_sequence=theta_sequence,
            drift_sequence=drift_sequence,
            scenario_type="flash_crash" if _is_flash_crash_preset(user_knobs) else None,
            delta_sequence=delta_sequence,
            user_knobs=user_knobs,
            return_metadata=True,
        )

        scenarios, scenario_metadata = scenarios

        metrics = garch.validate_scenarios(scenarios, user_knobs=user_knobs)

        all_rows = []
        for i, scenario_df in enumerate(scenarios):
            df = scenario_df.copy()
            df["scenario_id"] = i + 1
            all_rows.append(df)

        combined = pd.concat(all_rows, ignore_index=True)

        # Upload to S3
        csv_buffer = io.StringIO()
        combined.to_csv(csv_buffer, index=False)
        s3_key = f"garch/{job_id}.csv"

        s3_client.put_object(
            Bucket=S3_BUCKET_NAME, Key=s3_key, Body=csv_buffer.getvalue()
        )

        s3_url = f"s3://{S3_BUCKET_NAME}/{s3_key}"

        # Generate visualization plots
        # try:
        #     from worker.GARCH.services.visualization_service import VisualizationService

        #     viz_service = VisualizationService(
        #         data
        #     )  # Use 'data' (historical data downloaded above)

        #     # Plot 1: Price comparison
        #     price_plot_path = os.path.join(OUTPUT_DIR, f"{job_id}_prices.png")
        #     viz_service.plot_price_comparison(
        #         scenarios=scenarios,
        #         output_path=price_plot_path,
        #         title=f"Historical vs Synthetic Prices - {ticker}",
        #         num_scenarios_to_plot=50,
        #     )

        #     # Plot 2: Statistics comparison
        #     stats_plot_path = os.path.join(OUTPUT_DIR, f"{job_id}_stats.png")
        #     viz_service.plot_statistics_comparison(
        #         scenarios=scenarios,
        #         output_path=stats_plot_path,
        #         user_knobs=user_knobs,
        #         title=f"Synthetic vs Desired Characteristics - {ticker}",
        #     )

        #     logger.info(f"Generated plots: {price_plot_path}, {stats_plot_path}")
        # except Exception as e:
        #     logger.warning(f"Could not generate visualizations: {e}")

        # Compute chart data for frontend visualizations
        try:
            (
                selected_idx,
                selection_objective,
                selection_target,
                selection_value,
                selection_score,
                selection_breakdown,
            ) = _select_best_display_scenario(
                scenarios,
                user_knobs,
                historical_prices=data["Close"].values.astype(float),
                historical_returns=returns,
                scenario_metadata=scenario_metadata,
                delta_schedule=delta_sequence,
            )
            selected_idx = int(np.clip(selected_idx, 0, max(len(scenarios) - 1, 0)))

            # Recompute knob-level match metrics from the selected display path
            # (instead of pooled returns across all scenarios).
            try:
                from worker.GARCH.services.validation_service import ValidationService

                selected_returns = (
                    scenarios[selected_idx]["Close"].pct_change().dropna().values
                )
                validator = ValidationService(data)
                selected_metrics = validator.validate_against_desired(
                    selected_returns, user_knobs
                )

                for k in ["mean_match", "volatility_match", "kurtosis_match", "acf_match"]:
                    if k in selected_metrics:
                        metrics[k] = float(selected_metrics[k])

                metrics["overall_match"] = (
                    float(metrics.get("mean_match", 0.0))
                    + float(metrics.get("volatility_match", 0.0))
                    + float(metrics.get("kurtosis_match", 0.0))
                    + float(metrics.get("acf_match", 0.0))
                ) / 4.0
                metrics["match_metrics_source"] = "selected_path"
            except Exception as match_exc:
                logger.warning(
                    "Could not recompute selected-path match metrics for job %s: %s",
                    job_id,
                    match_exc,
                )
                metrics["match_metrics_source"] = "all_scenarios"

            logger.info(
                "Display path selection: objective=%s, selected_scenario=%s, target=%.8f, value=%.8f",
                selection_objective,
                selected_idx + 1,
                selection_target,
                selection_value,
            )

            # Fidelity metrics aligned with selection objective semantics:
            # - flash crash objectives: composite score itself (higher is better)
            # - other objectives: convert distance-style score to similarity
            # - csv_similarity: equal-weight similarity to input CSV stats
            if selection_objective in {"flash_crash_evaluation", "flash_crash_composite"}:
                intent_fidelity = float(np.clip(float(selection_score), 0.0, 1.0))
            else:
                intent_fidelity = float(np.clip(1.0 - float(selection_score), 0.0, 1.0))
            csv_similarity = intent_fidelity
            try:
                hist_r = np.asarray(returns, dtype=float).ravel()
                hist_r = hist_r[np.isfinite(hist_r)]
                sel_r = _get_log_returns_from_scenario(scenarios[selected_idx])
                sel_r = sel_r[np.isfinite(sel_r)]

                if len(hist_r) >= 2 and len(sel_r) >= 2:
                    hist_mean = float(np.mean(hist_r))
                    hist_std = float(np.std(hist_r, ddof=1)) if len(hist_r) > 1 else 0.0
                    hist_std = max(hist_std, 1e-6)
                    hist_hurst = _safe_hurst_momentum(hist_r)
                    hist_kurt = _safe_excess_kurtosis(hist_r)

                    sel_mean = float(np.mean(sel_r))
                    sel_std = float(np.std(sel_r, ddof=1)) if len(sel_r) > 1 else 0.0
                    sel_hurst = _safe_hurst_momentum(sel_r)
                    sel_kurt = _safe_excess_kurtosis(sel_r)

                    def _norm_err(actual: float, target: float, denom: float) -> float:
                        d = abs(float(actual) - float(target)) / max(float(denom), 1e-12)
                        return float(np.clip(d, 0.0, 1.0))

                    d_vol = _norm_err(sel_std, hist_std, hist_std)
                    d_trend = _norm_err(sel_mean, hist_mean, 0.5 * hist_std)
                    d_momentum = _norm_err(sel_hurst, hist_hurst, 0.5)
                    d_fat = _norm_err(sel_kurt, hist_kurt, max(abs(hist_kurt), 1.0))
                    avg_distance = float((d_vol + d_trend + d_momentum + d_fat) / 4.0)
                    csv_similarity = float(np.clip(1.0 - avg_distance, 0.0, 1.0))
            except Exception as fidelity_exc:
                logger.warning(
                    "Could not compute fidelity decomposition for job %s: %s",
                    job_id,
                    fidelity_exc,
                )

            chart_data = compute_chart_data(data, scenarios[selected_idx], all_scenarios=scenarios)
            chart_data["overallMatch"] = float(metrics.get("overall_match", 0.0))
            chart_data["selectedScenarioId"] = int(selected_idx + 1)
            chart_data["selectionObjective"] = selection_objective
            chart_data["selectionTarget"] = float(selection_target)
            chart_data["selectionValue"] = float(selection_value)
            chart_data["selectionScore"] = float(selection_score)
            chart_data["selectionBreakdown"] = selection_breakdown
            chart_data["intentFidelity"] = float(intent_fidelity)
            chart_data["csvSimilarity"] = float(csv_similarity)
            chart_data["desiredVolatility"] = float(user_knobs.get("desired_volatility", 1.0))

            # Keep kurtosis stats consistent with display-path selection basis:
            # use log-return excess kurtosis from input CSV and selected scenario path.
            try:
                hist_log_returns = np.asarray(returns, dtype=float).ravel()
                hist_log_returns = hist_log_returns[np.isfinite(hist_log_returns)]
                sel_log_returns = _get_log_returns_from_scenario(scenarios[selected_idx])
                sel_log_returns = sel_log_returns[np.isfinite(sel_log_returns)]

                chart_data["stats"]["historical"]["kurtosis"] = round(
                    float(_safe_excess_kurtosis(hist_log_returns)), 4
                )
                chart_data["stats"]["synthetic"]["kurtosis"] = round(
                    float(_safe_excess_kurtosis(sel_log_returns)), 4
                )
            except Exception as kurt_exc:
                logger.warning(
                    "Could not align chart kurtosis to selected path for job %s: %s",
                    job_id,
                    kurt_exc,
                )

            # Override per-scenario kurtosis/skewness/std with values computed
            # across all 100 scenarios (from validation), which are far more
            # statistically robust than the single-scenario estimates.
            if "skewness_synthetic" in metrics:
                chart_data["stats"]["synthetic"]["skewness"] = metrics[
                    "skewness_synthetic"
                ]
            if "skewness_historical" in metrics:
                chart_data["stats"]["historical"]["skewness"] = metrics[
                    "skewness_historical"
                ]
            if "volatility_synthetic" in metrics:
                chart_data["stats"]["synthetic"]["std"] = metrics[
                    "volatility_synthetic"
                ]
            if "volatility_historical" in metrics:
                chart_data["stats"]["historical"]["std"] = metrics[
                    "volatility_historical"
                ]

            job_store.set_chart_data(job_id, chart_data)
            logger.info("Chart data stored for job %s", job_id)
        except Exception as e:
            logger.warning("Could not compute chart data for job %s: %s", job_id, e, exc_info=True)

        # Persist results (now including predicted parameters)
        results_with_predictions = {
            **fitted_params,
            "delta_predicted": float(pred_params["delta"]),
            "theta_predicted": float(pred_params["theta"]),
            "delta_confidence": pred_params["delta_confidence"],
            "historical_hurst": float(user_knobs.get("historical_hurst", 0.5)),
            "target_hurst": float(user_knobs.get("target_hurst", 0.5)),
            "historical_skewness": float(skew_meta["historical_skewness"]),
            "historical_skewness_threshold": float(skew_meta["skew_threshold"]),
            "skew_detection_sample_size": int(skew_meta["sample_size"]),
            "use_skew_shocks_effective": bool(user_knobs["use_skew_shocks"]),
            "force_skewt_distribution_effective": bool(
                user_knobs["force_skewt_distribution"]
            ),
        }
        job_store.set_parameters(job_id, results_with_predictions)
        job_store.set_metrics(job_id, metrics)
        job_store.set_output_file(job_id, s3_url)
        job_store.set_status(job_id, "completed")

        # Update DB
        try:
            db = SessionLocal()
            update_job_status(db, job_id, JobStatus.completed, s3_url=s3_url)
            logger.info("Updated DB status for job %s", job_id)
        except Exception as db_exc:
            logger.error("Failed to update DB for job %s: %s", job_id, db_exc)
        finally:
            db.close()

        logger.info("Job %s completed successfully", job_id)

    except Exception as e:
        if job_id:
            logger.exception("Job %s failed", job_id)
            job_store.set_status(job_id, "failed")
            job_store.set_error(job_id, str(e))

            try:
                db = SessionLocal()
                update_job_status(db, job_id, JobStatus.failed)
                logger.info("Updated DB status to failed for job %s", job_id)
            except Exception as db_exc:
                logger.error(
                    "Failed to update DB for failed job %s: %s", job_id, db_exc
                )
            finally:
                db.close()
        else:
            logger.exception("Worker error before job pickup")
        time.sleep(1)
