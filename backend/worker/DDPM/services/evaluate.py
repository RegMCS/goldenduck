"""
evaluate.py
-----------
Evaluation service for DDPM synthetic OHLCV data.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import ks_2samp, kurtosis, skew, gaussian_kde
from worker.DDPM.utils import load_ohlcv


# ── helpers ────────────────────────────────────────────────────────────────

def _annualised_vol(returns: np.ndarray) -> float:
    return float(np.std(returns) * np.sqrt(252))

def _kurtosis(returns: np.ndarray) -> float:
    return float(kurtosis(returns, fisher=True))

def _momentum(returns: np.ndarray) -> float:
    return float(np.sum(returns[-252:])) if len(returns) >= 252 else float(np.sum(returns))

def _trend(returns: np.ndarray) -> float:
    cum = np.cumsum(returns)
    x   = np.arange(len(cum))
    slope, *_ = np.polyfit(x, cum, 1)
    return float(slope)

def _compute_stats(returns: np.ndarray) -> dict:
    return {
        "volatility": _annualised_vol(returns),
        "kurtosis"  : _kurtosis(returns),
        "momentum"  : _momentum(returns),
        "trend"     : _trend(returns),
        "skewness"  : float(skew(returns)),
        "mean"      : float(np.mean(returns)),
        "std"       : float(np.std(returns)),
    }


# ── main evaluation class ──────────────────────────────────────────────────

class EvaluationService:
    """
    Evaluate DDPM synthetic paths against real OHLCV data.

    Parameters
    ----------
    ohlcv_path : str
        Path to the real OHLCV CSV (same file used in finetune / inference).
    """

    def __init__(self, ohlcv_path: str = None):
        if ohlcv_path is not None:
            self.df_real, _ = load_ohlcv(ohlcv_path)
            self.real_returns = self.df_real["log_return"].values
            # Daily returns for plotting only
            self.real_daily_returns = np.expm1(self.real_returns)
        else:
            self.df_real             = None
            self.real_returns        = None
            self.real_daily_returns  = None


    # ── distribution + conditioning evaluation ───────────────────────────

    def evaluate(self,
                 synthetic_csv: str,
                 out_dir: str = "output/eval",
                 show_plot: bool = False) -> dict:
        """
        Full evaluation pipeline.

        Parameters
        ----------
        synthetic_csv : str
            Path to CSV produced by ReconstructionService.export_csv()
        out_dir : str
            Directory to save charts and summary CSV.
        show_plot : bool
            If True, display plots interactively (requires display).

        Returns
        -------
        dict : summary metrics
        """
        if self.real_returns is None:
            raise ValueError("ohlcv_path must be provided to EvaluationService for evaluate().")

        os.makedirs(out_dir, exist_ok=True)

        # Load synthetic paths
        df_syn  = pd.read_csv(synthetic_csv, parse_dates=["Date"], index_col="Date")
        n_paths = df_syn["path_id"].nunique()

        # Collect per-path log-returns
        syn_returns_list = []
        for pid, grp in df_syn.groupby("path_id"):
            r = np.log(grp["Close"] / grp["Close"].shift(1)).dropna().values
            syn_returns_list.append(r)

        syn_returns_all = np.concatenate(syn_returns_list)

        # Daily returns for plotting only (log → daily)
        syn_daily_returns_all = np.expm1(syn_returns_all)

        # ── stats ─────────────────────────────────────────────────────────
        real_stats = _compute_stats(self.real_returns)
        ks_stat, ks_pval = ks_2samp(self.real_returns, syn_returns_all)

        metrics_rows = [_compute_stats(r) for r in syn_returns_list]
        df_metrics   = pd.DataFrame(metrics_rows)

        # ── conditioning match scores (0–1) ───────────────────────────────
        target_keys = ["volatility", "kurtosis", "momentum", "trend"]
        match_scores = {}
        for k in target_keys:
            real_val = real_stats[k]
            syn_mean = df_metrics[k].mean()
            denom    = max(abs(real_val), 1e-8)
            err      = abs(syn_mean - real_val) / denom
            match_scores[k] = float(np.clip(1 - err, 0, 1))

        overall_quality = float(np.mean(list(match_scores.values())))

        summary = {
            "n_paths"        : n_paths,
            "ks_statistic"   : round(ks_stat,  4),
            "ks_pvalue"      : round(ks_pval,  4),
            "overall_quality": round(overall_quality, 4),
            **{f"real_{k}"   : round(real_stats[k], 6) for k in target_keys},
            **{f"syn_{k}"    : round(df_metrics[k].mean(), 6) for k in target_keys},
            **{f"match_{k}"  : round(match_scores[k], 4) for k in target_keys},
        }

        # ── print summary ─────────────────────────────────────────────────
        print("\n══════════════════════════════════════════")
        print("  DDPM Evaluation Summary")
        print("══════════════════════════════════════════")
        print(f"  Paths evaluated  : {n_paths}")
        print(f"  KS statistic     : {ks_stat:.4f}  (p={ks_pval:.4f})")
        print(f"  Overall quality  : {overall_quality:.4f}")
        print("\n  Conditioning Match Scores (0–1):")
        for k in target_keys:
            bar = "█" * int(match_scores[k] * 20)
            print(f"    {k:<12s}: {match_scores[k]:.3f}  |{bar:<20s}|")
        print("──────────────────────────────────────────")
        print(f"  {'Metric':<12s}  {'Real':>10s}  {'Synthetic':>10s}")
        for k in target_keys:
            print(f"  {k:<12s}  {real_stats[k]:>10.5f}  {df_metrics[k].mean():>10.5f}")
        print("══════════════════════════════════════════\n")

        # ── charts ────────────────────────────────────────────────────────
        self._plot_return_distribution(
            syn_returns_all, syn_daily_returns_all, out_dir, show_plot)
        self._plot_conditioning_match(
            real_stats, df_metrics, match_scores, out_dir, show_plot)
        self._plot_quality_dashboard(
            syn_returns_list, real_stats, match_scores,
            overall_quality, out_dir, show_plot)

        # ── save CSVs ─────────────────────────────────────────────────────
        pd.DataFrame([summary]).to_csv(os.path.join(out_dir, "eval_summary.csv"), index=False)
        df_metrics.to_csv(os.path.join(out_dir, "per_path_metrics.csv"), index=False)
        print(f"Results saved to {out_dir}/")

        return summary


    # ── chart helpers ─────────────────────────────────────────────────────

    def _plot_return_distribution(self, syn_returns_all, syn_daily_returns_all,
                                   out_dir, show_plot):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("Return Distribution: Real vs Synthetic",
                     fontsize=14, fontweight="bold")

        # ── Histogram + KDE (daily returns on x-axis) ─────────────────────
        ax = axes[0]
        real_daily = self.real_daily_returns
        syn_daily  = syn_daily_returns_all

        bins = np.linspace(
            min(real_daily.min(), syn_daily.min()),
            max(real_daily.max(), syn_daily.max()), 80)

        ax.hist(real_daily, bins=bins, alpha=0.6, color="#2196F3",
                density=True, label="Real")
        ax.hist(syn_daily,  bins=bins, alpha=0.6, color="#FF5722",
                density=True, label="Synthetic")

        xr = np.linspace(bins[0], bins[-1], 300)
        ax.plot(xr, gaussian_kde(real_daily)(xr), color="#1565C0", lw=2)
        ax.plot(xr, gaussian_kde(syn_daily)(xr),  color="#BF360C", lw=2)

        ax.set_xlabel("Daily Return")   # ← changed from "Log Return"
        ax.set_ylabel("Density")
        ax.set_title("Histogram + KDE")
        ax.legend()

        # ── QQ plot (still uses log returns for statistical accuracy) ──────
        ax = axes[1]
        real_q = np.quantile(self.real_returns, np.linspace(0.01, 0.99, 200))
        syn_q  = np.quantile(syn_returns_all,   np.linspace(0.01, 0.99, 200))
        ax.scatter(real_q, syn_q, s=10, alpha=0.6, color="#7B1FA2")
        lim = max(abs(real_q).max(), abs(syn_q).max())
        ax.plot([-lim, lim], [-lim, lim], "k--", lw=1, label="Perfect match")
        ax.set_xlabel("Real Quantiles")
        ax.set_ylabel("Synthetic Quantiles")
        ax.set_title("QQ Plot")
        ax.legend()

        plt.tight_layout()
        fig.savefig(os.path.join(out_dir, "return_distribution.png"), dpi=150)
        if show_plot: plt.show()
        plt.close(fig)


    def _plot_conditioning_match(self, real_stats, df_metrics, match_scores,
                                  out_dir, show_plot):
        target_keys = ["volatility", "kurtosis", "momentum", "trend"]
        colors      = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63"]

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Synthetic vs Real: Conditioning Metrics",
                     fontsize=14, fontweight="bold")

        for ax, key, col in zip(axes.flat, target_keys, colors):
            vals = df_metrics[key].values
            ax.hist(vals, bins=30, color=col, alpha=0.75, edgecolor="white")
            ax.axvline(real_stats[key], color="black", lw=2, linestyle="--",
                       label=f"Real: {real_stats[key]:.4f}")
            ax.axvline(vals.mean(), color="red", lw=2, linestyle="-",
                       label=f"Syn mean: {vals.mean():.4f}")
            ax.set_title(f"{key.capitalize()}  (match={match_scores[key]:.3f})")
            ax.set_xlabel(key.capitalize())
            ax.set_ylabel("Count")
            ax.legend(fontsize=8)

        plt.tight_layout()
        fig.savefig(os.path.join(out_dir, "conditioning_match.png"), dpi=150)
        if show_plot: plt.show()
        plt.close(fig)


    def _plot_quality_dashboard(self, syn_returns_list, real_stats,
                                 match_scores, overall_quality,
                                 out_dir, show_plot):
        target_keys = ["volatility", "kurtosis", "momentum", "trend"]
        colors      = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63"]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f"Synthetic Data Quality  (Overall: {overall_quality:.3f})",
                     fontsize=14, fontweight="bold")

        # Match score bar chart
        ax = axes[0]
        bars = ax.barh(target_keys,
                       [match_scores[k] for k in target_keys],
                       color=colors, edgecolor="white", height=0.5)
        ax.axvline(0.8, color="green",  lw=1.5, linestyle="--", label="Good (0.8)")
        ax.axvline(0.6, color="orange", lw=1.5, linestyle="--", label="Fair (0.6)")
        ax.set_xlim(0, 1.05)
        ax.set_xlabel("Match Score (0–1)")
        ax.set_title("Conditioning Match Scores")
        ax.legend(fontsize=8)
        for bar, k in zip(bars, target_keys):
            ax.text(bar.get_width() + 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    f"{match_scores[k]:.3f}", va="center", fontsize=9)

        # Sample synthetic paths vs real tail
        ax = axes[1]
        sample_paths = syn_returns_list[:min(20, len(syn_returns_list))]
        for r in sample_paths:
            ax.plot(np.cumsum(r), alpha=0.3, lw=0.8, color="#FF5722")
        ax.plot(np.cumsum(self.real_returns[-len(sample_paths[0]):]),
                color="black", lw=2, label="Real (tail)")
        ax.set_xlabel("Trading Days")
        ax.set_ylabel("Cumulative Log Return")
        ax.set_title("Synthetic Paths vs Real Tail")
        ax.legend()

        plt.tight_layout()
        fig.savefig(os.path.join(out_dir, "quality_dashboard.png"), dpi=150)
        if show_plot: plt.show()
        plt.close(fig)

        print(f"Charts saved:\n"
              f"  {out_dir}/return_distribution.png\n"
              f"  {out_dir}/conditioning_match.png\n"
              f"  {out_dir}/quality_dashboard.png")


    # ── parameter sensitivity (existing method — unchanged) ───────────────

    def plot_parameter_sensitivity(self, svc,
                                    n_paths: int   = 20,
                                    n_display: int = 5,
                                    out_path: str  = "parameter_sensitivity.png"):
        """
        Generate 4 parameter sensitivity panels.
        Each panel shows synthetic paths with one parameter varied,
        all others held at base (calm) values.

        svc       : InferenceService instance
        n_paths   : paths to generate per condition
        n_display : paths to draw per subplot (keep low for clarity)
        out_path  : output PNG path
        """
        BASE = {
            "realised_vol" : 0.15,
            "drift"        : 0.00,
            "tail_index"   : 0.010,
            "momentum"     : 0.00,
        }

        def gen(overrides: dict) -> np.ndarray:
            params = {**BASE, **overrides}
            return svc.generate(
                regime="custom",
                custom_cond=params,
                n_paths=n_paths
            )

        def to_price(paths: np.ndarray, start: float = 100.0) -> np.ndarray:
            return start * np.exp(np.cumsum(paths, axis=1))

        def plot_paths(ax, paths: np.ndarray, color: str,
                       title: str, label_params: dict,
                       show_acf_inset: bool = False):
            prices = to_price(paths)
            days   = np.arange(prices.shape[1])
            for i in range(min(n_display, len(prices))):
                ax.plot(days, prices[i], color=color, alpha=0.35, linewidth=0.8)
            ax.plot(days, prices.mean(axis=0), color=color,
                    linewidth=2.0, label="Mean path")
            ax.set_title(title, fontsize=10, fontweight="bold")
            ax.set_xlabel("Trading Days", fontsize=8)
            ax.set_ylabel("Price", fontsize=8)
            ax.tick_params(labelsize=7)
            param_str = "\n".join([f"{k}={v:.3f}" for k, v in label_params.items()])
            ax.text(0.03, 0.97, param_str,
                    transform=ax.transAxes,
                    fontsize=6.5, verticalalignment="top",
                    bbox=dict(boxstyle="round,pad=0.3",
                              facecolor="white", alpha=0.7))

        fig = plt.figure(figsize=(20, 22))
        gs  = gridspec.GridSpec(5, 3, figure=fig, hspace=0.45, wspace=0.35)
        fig.suptitle("Parameter Sensitivity — Synthetic Path Transformations",
                     fontsize=14, fontweight="bold", y=0.98)

        colors = {
            "neutral" : "#4e79a7",
            "low"     : "#59a14f",
            "mid"     : "#f28e2b",
            "high"    : "#e15759",
            "negative": "#b07aa1",
        }

        # ROW 1 — MOMENTUM
        momentum_cases = [
            (-0.30, "Strong Mean-Reversion\n(momentum = −0.30)", colors["negative"]),
            ( 0.00, "No Momentum\n(momentum = 0.00)",            colors["neutral"]),
            ( 0.30, "Strong Momentum\n(momentum = +0.30)",       colors["high"]),
        ]
        for col, (mom_val, title, color) in enumerate(momentum_cases):
            ax = fig.add_subplot(gs[0, col])
            paths = gen({"momentum": mom_val})
            plot_paths(ax, paths, color, title, label_params={"momentum": mom_val})

        # ACF inset
        ax_acf = fig.add_subplot(gs[1, :])
        from statsmodels.tsa.stattools import acf as _acf
        ax_acf.set_title("ACF of Returns by Momentum Level", fontsize=10)
        for mom_val, label, color in momentum_cases:
            paths = gen({"momentum": mom_val})
            acf_vals = _acf(paths.flatten(), nlags=40, fft=True)
            ax_acf.plot(acf_vals, color=color, linewidth=1.5,
                        label=f"momentum={mom_val:+.2f}")
        ax_acf.axhline(0, color="black", linewidth=0.5)
        ax_acf.set_xlabel("Lag", fontsize=8)
        ax_acf.set_ylabel("Autocorrelation", fontsize=8)
        ax_acf.legend(fontsize=8)
        ax_acf.tick_params(labelsize=7)

        # ROW 2 — TREND (DRIFT)
        drift_cases = [
            (-0.40, "Strong Downtrend\n(drift = −0.40)", colors["high"]),
            ( 0.00, "No Trend\n(drift = 0.00)",           colors["neutral"]),
            ( 0.40, "Strong Uptrend\n(drift = +0.40)",    colors["low"]),
        ]
        for col, (drift_val, title, color) in enumerate(drift_cases):
            ax = fig.add_subplot(gs[2, col])
            paths = gen({"drift": drift_val})
            plot_paths(ax, paths, color, title, label_params={"drift": drift_val})

        # ROW 3 — VOLATILITY
        vol_cases = [
            (0.10, "Calm\n(vol = 0.10)",     colors["low"]),
            (0.30, "High-Vol\n(vol = 0.30)", colors["mid"]),
            (0.55, "Crisis\n(vol = 0.55)",   colors["high"]),
        ]
        for col, (vol_val, title, color) in enumerate(vol_cases):
            ax = fig.add_subplot(gs[3, col])
            paths = gen({"realised_vol": vol_val})
            plot_paths(ax, paths, color, title, label_params={"realised_vol": vol_val})

        # ROW 4 — TAIL INDEX
        tail_cases = [
            (0.005, "Thin Tails\n(tail = 0.005)", colors["low"]),
            (0.030, "Medium Tails\n(tail = 0.030)", colors["mid"]),
            (0.080, "Fat Tails\n(tail = 0.080)",   colors["high"]),
        ]
        for col, (tail_val, title, color) in enumerate(tail_cases):
            ax = fig.add_subplot(gs[4, col])
            paths = gen({"tail_index": tail_val})
            plot_paths(ax, paths, color, title, label_params={"tail_index": tail_val})

        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Parameter sensitivity plot saved → {out_path}")