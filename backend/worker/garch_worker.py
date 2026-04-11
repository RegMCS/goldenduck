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

    return {
        "mean": round(mean_r, 6),
        "std": round(std_r, 6),
        "skewness": round(float(scipy_stats.skew(returns_arr)), 4),
        "kurtosis": round(float(scipy_stats.kurtosis(returns_arr)), 4),
        "maxDrawdown": round(max_dd, 6),
        "sharpe": round(sharpe, 4),
        "annualizedReturn": round(ann_return, 6),
        "annualizedVol": round(ann_vol, 6),
        "totalReturn": round(total_return, 6),
        "numDataPoints": n,
    }


def compute_chart_data(historical_df: pd.DataFrame, scenario: pd.DataFrame) -> dict:
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

    return {
        "historical": historical,
        "synthetic": synthetic,
        "timeSeries": time_series,
        "returns": returns_data,
        "drawdowns": drawdowns,
        "stats": stats,
    }


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
            "desired_volatility": float(params.get("desired_volatility", 1.0)),
            "desired_trend": float(params.get("desired_trend", 0.0)),
            "desired_fat_tails": float(params.get("desired_fat_tails", 1.0)),
            "desired_momentum": float(params.get("desired_momentum", 0.5)),
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

        garch = GARCHService()

        fitted_params = garch.fit_with_retry(data, p=p, q=q)

        # Predict delta and theta using ML
        logger.info(f"Predicting GARCH-FX parameters from user knobs...")
        returns = np.log(data["Close"].values[1:] / data["Close"].values[:-1])

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

        logger.info(f"  Delta (ML): {pred_params['delta']:.4f}")
        logger.info(f"  Theta (heuristic): {pred_params['theta']:.6f}")

        delta_sequence = np.full(horizon, float(pred_params["delta"]))
        scenarios = garch.generate_scenarios_fx(
            num_scenarios=num_scenarios,
            horizon=horizon,
            theta=float(pred_params["theta"]),
            delta_sequence=delta_sequence,
            user_knobs=user_knobs,
        )

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
        try:
            from worker.GARCH.services.visualization_service import VisualizationService

            viz_service = VisualizationService(
                data
            )  # Use 'data' (historical data downloaded above)

            # Plot 1: Price comparison
            price_plot_path = os.path.join(OUTPUT_DIR, f"{job_id}_prices.png")
            viz_service.plot_price_comparison(
                scenarios=scenarios,
                output_path=price_plot_path,
                title=f"Historical vs Synthetic Prices - {ticker}",
                num_scenarios_to_plot=50,
            )

            # Plot 2: Statistics comparison
            stats_plot_path = os.path.join(OUTPUT_DIR, f"{job_id}_stats.png")
            viz_service.plot_statistics_comparison(
                scenarios=scenarios,
                output_path=stats_plot_path,
                user_knobs=user_knobs,
                title=f"Synthetic vs Desired Characteristics - {ticker}",
            )

            logger.info(f"Generated plots: {price_plot_path}, {stats_plot_path}")
        except Exception as e:
            logger.warning(f"Could not generate visualizations: {e}")

        # Compute chart data for frontend visualizations
        try:
            chart_data = compute_chart_data(data, scenarios[0])
            chart_data["overallMatch"] = float(metrics.get("overall_match", 0.0))

            # Override per-scenario kurtosis/skewness/std with values computed
            # across all 100 scenarios (from validation), which are far more
            # statistically robust than the single-scenario estimates.
            if "kurtosis_synthetic" in metrics:
                chart_data["stats"]["synthetic"]["kurtosis"] = metrics[
                    "kurtosis_synthetic"
                ]
            if "kurtosis_historical" in metrics:
                chart_data["stats"]["historical"]["kurtosis"] = metrics[
                    "kurtosis_historical"
                ]
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
            logger.warning("Could not compute chart data for job %s: %s", job_id, e)

        # Persist results (now including predicted parameters)
        results_with_predictions = {
            **fitted_params,
            "delta_predicted": float(pred_params["delta"]),
            "theta_predicted": float(pred_params["theta"]),
            "delta_confidence": pred_params["delta_confidence"],
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
