# services/visualization_service.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional
import logging
import os

logger = logging.getLogger(__name__)

# Output directory for visualizations
OUTPUT_DIR = os.path.abspath("/app/output")
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR, exist_ok=True)


class VisualizationService:
    """
    Service for visualizing historical vs synthetic data
    """

    def __init__(self, historical_data: pd.DataFrame):
        self.historical_data = historical_data

    def plot_price_comparison(
        self,
        scenarios: List[pd.DataFrame],
        output_path: str,
        title: str = "Historical vs Synthetic Price Paths",
        num_scenarios_to_plot: int = 50,
    ) -> None:
        """
        Plot historical close prices against generated synthetic scenarios.

        Args:
            scenarios: List of DataFrames containing synthetic OHLCV data
            output_path: Path to save the plot
            title: Plot title
            num_scenarios_to_plot: Number of synthetic paths to display
        """
        try:
            # Setup figure
            plt.figure(figsize=(14, 8))

            # Plot historical data
            historical_close = self.historical_data["Close"].values
            historical_dates = np.arange(len(historical_close))
            plt.plot(
                historical_dates,
                historical_close,
                "k-",
                linewidth=2.5,
                label="Historical",
                zorder=10,
            )

            # Plot synthetic scenarios (subset for clarity)
            num_to_plot = min(num_scenarios_to_plot, len(scenarios))
            colors = sns.color_palette("Blues", n_colors=num_to_plot)

            for idx in range(num_to_plot):
                scenario_close = scenarios[idx]["Close"].values
                scenario_dates = np.arange(
                    len(historical_close), len(historical_close) + len(scenario_close)
                )

                alpha = 0.3 if idx > 0 else 0.5  # First scenario slightly more opaque
                plt.plot(
                    scenario_dates,
                    scenario_close,
                    color=colors[idx],
                    alpha=alpha,
                    linewidth=1.0,
                    label="Synthetic" if idx == 0 else "",
                )

            # Add vertical line at forecast boundary
            boundary = len(historical_close)
            plt.axvline(
                x=boundary,
                color="red",
                linestyle="--",
                linewidth=1.5,
                alpha=0.7,
                label="Forecast Start",
            )

            # Labels and formatting
            plt.xlabel("Trading Days", fontsize=12)
            plt.ylabel("Price ($)", fontsize=12)
            plt.title(title, fontsize=14, fontweight="bold")
            plt.legend(loc="best", fontsize=11)
            plt.grid(True, alpha=0.3)

            # Save
            plt.tight_layout()
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            plt.close()

            logger.info(f"Saved price comparison plot to {output_path}")

        except Exception as e:
            logger.error(f"Error plotting price comparison: {e}", exc_info=True)

    def plot_statistics_comparison(
        self,
        scenarios: List[pd.DataFrame],
        output_path: str,
        user_knobs: Optional[dict] = None,
        title: str = "Synthetic vs Desired Characteristics",
    ) -> None:
        """
        Plot distribution statistics of synthetic returns vs desired knobs.

        Args:
            scenarios: List of DataFrames containing synthetic OHLCV data
            output_path: Path to save the plot
            user_knobs: User knobs specifying desired characteristics
            title: Plot title
        """
        try:
            from GARCH.services.validation_service import ValidationService

            # Extract synthetic returns
            all_returns = []
            for scenario in scenarios:
                returns = scenario["Close"].pct_change().dropna().values
                all_returns.extend(returns)

            synthetic_returns = np.array(all_returns)
            historical_returns = (
                self.historical_data["Close"].pct_change().dropna().values
            )

            # Ensure 1D arrays
            synthetic_returns = np.asarray(synthetic_returns).flatten()
            historical_returns = np.asarray(historical_returns).flatten()

            # Create subplots
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            fig.suptitle(title, fontsize=14, fontweight="bold")

            # 1. Return distributions
            ax = axes[0, 0]
            ax.hist(
                historical_returns * 100,
                bins=50,
                alpha=0.6,
                label="Historical",
                density=True,
                color="black",
            )
            ax.hist(
                synthetic_returns * 100,
                bins=50,
                alpha=0.6,
                label="Synthetic",
                density=True,
                color="blue",
            )
            ax.set_xlabel("Daily Returns (%)", fontsize=11)
            ax.set_ylabel("Density", fontsize=11)
            ax.set_title("Return Distribution", fontsize=12, fontweight="bold")
            ax.legend()
            ax.grid(True, alpha=0.3)

            # 2. Volatility (rolling 20-day)
            ax = axes[0, 1]
            hist_vol = pd.Series(historical_returns).rolling(window=20).std() * 100
            ax.plot(hist_vol.values, label="Historical", linewidth=2, color="black")
            # For synthetic, show mean and bands
            synth_vol = pd.Series(synthetic_returns).rolling(window=20).std() * 100
            ax.plot(
                [synth_vol.mean()] * len(hist_vol),
                label="Synthetic (Mean)",
                linewidth=2,
                color="blue",
                linestyle="--",
            )
            ax.fill_between(
                range(len(hist_vol)),
                synth_vol.mean() - synth_vol.std(),
                synth_vol.mean() + synth_vol.std(),
                alpha=0.2,
                color="blue",
                label="Synthetic (±1σ)",
            )
            ax.set_xlabel("Window", fontsize=11)
            ax.set_ylabel("Volatility (%)", fontsize=11)
            ax.set_title("Rolling 20-Day Volatility", fontsize=12, fontweight="bold")
            ax.legend()
            ax.grid(True, alpha=0.3)

            # 3. Kurtosis comparison
            ax = axes[1, 0]
            from scipy import stats

            metrics = {
                "Kurtosis": [
                    stats.kurtosis(historical_returns),
                    stats.kurtosis(synthetic_returns),
                ],
                "Skewness": [
                    stats.skew(historical_returns),
                    stats.skew(synthetic_returns),
                ],
                "Volatility": [
                    np.std(historical_returns) * 100,
                    np.std(synthetic_returns) * 100,
                ],
            }

            x = np.arange(len(metrics))
            width = 0.35
            hist_vals = [metrics[k][0] for k in metrics]
            synth_vals = [metrics[k][1] for k in metrics]

            bars1 = ax.bar(
                x - width / 2,
                hist_vals,
                width,
                label="Historical",
                color="black",
                alpha=0.7,
            )
            bars2 = ax.bar(
                x + width / 2,
                synth_vals,
                width,
                label="Synthetic",
                color="blue",
                alpha=0.7,
            )

            ax.set_ylabel("Value", fontsize=11)
            ax.set_title("Distribution Metrics", fontsize=12, fontweight="bold")
            ax.set_xticks(x)
            ax.set_xticklabels(metrics.keys())
            ax.legend()
            ax.grid(True, alpha=0.3, axis="y")

            # 4. User knobs (if provided)
            ax = axes[1, 1]
            if user_knobs:
                knob_names = list(user_knobs.keys())
                knob_values = list(user_knobs.values())
                colors_knobs = [
                    "green" if v > 1.0 else "orange" if v < 1.0 else "gray"
                    for v in knob_values
                ]
                bars = ax.barh(knob_names, knob_values, color=colors_knobs, alpha=0.7)
                ax.axvline(
                    x=1.0,
                    color="red",
                    linestyle="--",
                    linewidth=2,
                    label="Baseline (1.0)",
                )
                ax.set_xlabel("Value", fontsize=11)
                ax.set_title("User Knobs", fontsize=12, fontweight="bold")
                ax.legend()
                ax.grid(True, alpha=0.3, axis="x")
            else:
                ax.text(
                    0.5,
                    0.5,
                    "No user knobs provided",
                    ha="center",
                    va="center",
                    fontsize=12,
                )
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)

            # Save
            plt.tight_layout()
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            plt.close()

            logger.info(f"Saved statistics comparison plot to {output_path}")

        except Exception as e:
            logger.error(f"Error plotting statistics comparison: {e}", exc_info=True)
