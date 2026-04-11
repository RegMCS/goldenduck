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
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import numpy as np

    # ── Base conditioning (neutral / calm) ────────────────────────────────
    BASE = {
        "realised_vol" : 0.15,
        "drift"        : 0.00,
        "tail_index"   : 0.010,
        "momentum"     : 0.00,
    }

    def gen(overrides: dict) -> np.ndarray:
        """Generate n_paths with base params + overrides."""
        params = {**BASE, **overrides}
        return svc.generate(
            regime="custom",
            custom_cond=params,
            n_paths=n_paths
        )   # (n_paths, seq_len)

    def to_price(paths: np.ndarray, start: float = 100.0) -> np.ndarray:
        """Convert log-returns to price paths."""
        return start * np.exp(np.cumsum(paths, axis=1))

    def plot_paths(ax, paths: np.ndarray, color: str,
                   title: str, label_params: dict,
                   show_acf_inset: bool = False):
        prices = to_price(paths)
        days   = np.arange(prices.shape[1])

        # Draw individual paths
        for i in range(min(n_display, len(prices))):
            ax.plot(days, prices[i], color=color, alpha=0.35,
                    linewidth=0.8)

        # Draw mean path bold
        ax.plot(days, prices.mean(axis=0), color=color,
                linewidth=2.0, label="Mean path")

        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xlabel("Trading Days", fontsize=8)
        ax.set_ylabel("Price", fontsize=8)
        ax.tick_params(labelsize=7)

        # Param annotation box
        param_str = "\n".join([f"{k}={v:.3f}" for k, v in label_params.items()])
        ax.text(0.03, 0.97, param_str,
                transform=ax.transAxes,
                fontsize=6.5, verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.3",
                          facecolor="white", alpha=0.7))

    # ── Figure layout: 4 rows, 3 cols max ────────────────────────────────
    fig = plt.figure(figsize=(20, 22))
    gs  = gridspec.GridSpec(5, 3, figure=fig,
                            hspace=0.45, wspace=0.35)
    fig.suptitle("Parameter Sensitivity — Synthetic Path Transformations",
                 fontsize=14, fontweight="bold", y=0.98)

    colors = {
        "neutral" : "#4e79a7",
        "low"     : "#59a14f",
        "mid"     : "#f28e2b",
        "high"    : "#e15759",
        "negative": "#b07aa1",
    }

    # ════════════════════════════════════════════════════════════════════
    # ROW 1 — MOMENTUM
    # ════════════════════════════════════════════════════════════════════
    momentum_cases = [
        (-0.30, "Strong Mean-Reversion\n(momentum = −0.30)", colors["negative"]),
        ( 0.00, "No Momentum\n(momentum = 0.00)",            colors["neutral"]),
        ( 0.30, "Strong Momentum\n(momentum = +0.30)",       colors["high"]),
    ]
    for col, (mom_val, title, color) in enumerate(momentum_cases):
        ax = fig.add_subplot(gs[0, col])
        paths = gen({"momentum": mom_val})
        plot_paths(ax, paths, color, title,
                   label_params={"momentum": mom_val})

    # ── Row 1 inset: ACF of returns for each momentum case ────────────
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

    # ════════════════════════════════════════════════════════════════════
    # ROW 2 — TREND (DRIFT)
    # ════════════════════════════════════════════════════════════════════
    drift_cases = [
        (-0.40, "Strong Downtrend\n(drift = −0.40)", colors["high"]),
        ( 0.00, "No Trend\n(drift = 0.00)",           colors["neutral"]),
        ( 0.40, "Strong Uptrend\n(drift = +0.40)",    colors["low"]),
    ]
    for col, (drift_val, title, color) in enumerate(drift_cases):
        ax = fig.add_subplot(gs[2, col])
        paths = gen({"drift": drift_val})
        plot_paths(ax, paths, color, title,
                   label_params={"drift": drift_val})

    # ════════════════════════════════════════════════════════════════════
    # ROW 3 — VOLATILITY
    # ════════════════════════════════════════════════════════════════════
    vol_cases = [
        (0.10, "Calm\n(vol = 0.10)",     colors["low"]),
        (0.30, "High-Vol\n(vol = 0.30)", colors["mid"]),
        (0.55, "Crisis\n(vol = 0.55)",   colors["high"]),
    ]
    for col, (vol_val, title, color) in enumerate(vol_cases):
        ax = fig.add_subplot(gs[3, col])
        paths = gen({"realised_vol": vol_val})
        plot_paths(ax, paths, color, title,
                   label_params={"realised_vol": vol_val})

    # ════════════════════════════════════════════════════════════════════
    # ROW 4 — TAIL INDEX
    # ════════════════════════════════════════════════════════════════════
    tail_cases = [
        (0.005, "Thin Tails\n(tail = 0.005)", colors["low"]),
        (0.030, "Medium Tails\n(tail = 0.030)", colors["mid"]),
        (0.080, "Fat Tails\n(tail = 0.080)",  colors["high"]),
    ]
    for col, (tail_val, title, color) in enumerate(tail_cases):
        ax = fig.add_subplot(gs[4, col])
        paths = gen({"tail_index": tail_val})
        plot_paths(ax, paths, color, title,
                   label_params={"tail_index": tail_val})

    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Parameter sensitivity plot saved → {out_path}")