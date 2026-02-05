"""
Visualize model evaluation results
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (15, 10)

# Load results
SCRIPT_DIR = Path(__file__).parent
ML_TRAINING_DIR = SCRIPT_DIR.parent
EVALUATION_DIR = ML_TRAINING_DIR / "models" / "evaluation"

with open(EVALUATION_DIR / "evaluation_report.json", "r") as f:
    report = json.load(f)

# Create figure with subplots
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle("Model Evaluation Results", fontsize=16, fontweight="bold")

# ============================================================
# 1. R² Comparison
# ============================================================
ax = axes[0, 0]
r2_scores = {
    "Delta": report["direct_metrics"]["delta"]["r2"],
}
colors = [
    "red" if v < 0.3 else "orange" if v < 0.7 else "green" for v in r2_scores.values()
]
bars = ax.bar(r2_scores.keys(), r2_scores.values(), color=colors, alpha=0.7)
ax.axhline(y=0.7, color="green", linestyle="--", label="Good (>0.7)")
ax.axhline(y=0.3, color="orange", linestyle="--", label="Acceptable (>0.3)")
ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
ax.set_ylabel("R² Score")
ax.set_title("R² Scores (Higher is Better)")
ax.legend()
ax.set_ylim(-0.3, 1.0)

# Add value labels
for bar in bars:
    height = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2.0,
        height,
        f"{height:.3f}",
        ha="center",
        va="bottom" if height > 0 else "top",
    )

# ============================================================
# 2. RMSE Comparison
# ============================================================
ax = axes[0, 1]
rmse_scores = {
    "Delta": report["direct_metrics"]["delta"]["rmse"],
}
ax.bar(rmse_scores.keys(), rmse_scores.values(), color=["coral"], alpha=0.7)
ax.set_ylabel("RMSE")
ax.set_title("Root Mean Square Error (Lower is Better)")

for i, (k, v) in enumerate(rmse_scores.items()):
    ax.text(i, v, f"{v:.4f}", ha="center", va="bottom")

# ============================================================
# 3. Baseline Comparison
# ============================================================
ax = axes[0, 2]
baseline_data = report["baseline_comparison"]
methods = ["Fixed\nParams", "Simple\nHeuristic", "AI\nModel"]
scores = [
    baseline_data["baseline_fixed"],
    baseline_data["baseline_heuristic"],
    baseline_data["ai_score"],
]
colors_base = ["gray", "orange", "blue"]
bars = ax.bar(methods, scores, color=colors_base, alpha=0.7)
ax.set_ylabel("Quality Score")
ax.set_title("Baseline Comparison (Higher is Better)")
ax.set_ylim(0, 1)

# Add value labels
for bar in bars:
    height = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2.0,
        height,
        f"{height:.3f}",
        ha="center",
        va="bottom",
    )

# Highlight winner
best_idx = np.argmax(scores)
bars[best_idx].set_edgecolor("green")
bars[best_idx].set_linewidth(3)

# ============================================================
# 4. MAPE Comparison
# ============================================================
ax = axes[1, 0]
mape_scores = {
    "Delta": min(
        report["direct_metrics"]["delta"]["mape"], 100
    ),  # Cap at 100% for visualization
}
bars = ax.bar(mape_scores.keys(), mape_scores.values(), color=["red"], alpha=0.7)
ax.axhline(y=20, color="green", linestyle="--", label="Good (<20%)")
ax.set_ylabel("MAPE (%)")
ax.set_title("Mean Absolute Percentage Error (Lower is Better)")
ax.legend()

for bar, (k, v) in zip(bars, mape_scores.items()):
    actual_mape = report["direct_metrics"]["delta"]["mape"]
    height = bar.get_height()
    label = (
        f"{actual_mape:.1f}%"
        if actual_mape < 1000
        else f"{actual_mape:.0f}%\n(VERY BAD)"
    )
    ax.text(
        bar.get_x() + bar.get_width() / 2.0,
        height,
        label,
        ha="center",
        va="bottom",
        fontsize=9,
    )

# ============================================================
# 5. End-to-End Quality
# ============================================================
ax = axes[1, 1]
e2e_score = report["end_to_end_metrics"]["mean_score"]
colors_e2e = ["red" if e2e_score < 0.5 else "orange" if e2e_score < 0.7 else "green"]
ax.barh(["Quality\nScore"], [e2e_score], color=colors_e2e, alpha=0.7)
ax.axvline(x=0.7, color="green", linestyle="--", linewidth=2, label="Target (0.7)")
ax.axvline(x=0.5, color="orange", linestyle="--", linewidth=2, label="Minimum (0.5)")
ax.set_xlim(0, 1)
ax.set_xlabel("Score")
ax.set_title("End-to-End Synthetic Data Quality")
ax.legend()
ax.text(e2e_score, 0, f"  {e2e_score:.3f}", va="center", fontsize=12, fontweight="bold")

# ============================================================
# 6. Pass/Fail Summary
# ============================================================
ax = axes[1, 2]
ax.axis("off")

# Summary text
checks = report["quality_checks"]
overall = report["overall_pass"]

summary_text = "EVALUATION SUMMARY\n" + "=" * 40 + "\n\n"
summary_text += f"Dataset Size:\n"
summary_text += f"  Train: {report['dataset_size']['train']}\n"
summary_text += f"  Val:   {report['dataset_size']['val']}\n"
summary_text += f"  Test:  {report['dataset_size']['test']}\n\n"

summary_text += "Quality Checks:\n"
for check, passed in checks.items():
    status = "✓" if passed else "✗"
    summary_text += f"  {status} {check}\n"

summary_text += f"\nOverall: {'PASS ✓' if overall else 'FAIL ✗'}\n\n"

summary_text += "Key Issues:\n"
summary_text += "  • Tiny dataset (340 train samples)\n"
summary_text += "  • Poor feature-to-sample ratio\n"
summary_text += "  • AI worse than baselines\n"
summary_text += "  • Negative R² for theta\n"

ax.text(
    0.1,
    0.5,
    summary_text,
    transform=ax.transAxes,
    fontsize=10,
    verticalalignment="center",
    fontfamily="monospace",
    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
)

plt.tight_layout()
plt.savefig(EVALUATION_DIR / "evaluation_dashboard.png", dpi=300, bbox_inches="tight")
print(f"✓ Dashboard saved to {EVALUATION_DIR / 'evaluation_dashboard.png'}")
plt.show()


# ============================================================
# Create detailed performance report
# ============================================================
print("\n" + "=" * 80)
print("DETAILED PERFORMANCE ANALYSIS")
print("=" * 80)

print("\n📊 Dataset Statistics:")
print(f"  Train samples: {report['dataset_size']['train']}")
print(f"  Samples per feature: {report['dataset_size']['train'] / 33:.1f}")
print(f"  Recommended minimum: 3,300 (100 per feature)")
print(f"  Status: ❌ INSUFFICIENT DATA")

print("\n📈 Delta Parameter:")
print(f"  R²:   {report['direct_metrics']['delta']['r2']:.3f} (target: >0.7) ❌")
print(f"  RMSE: {report['direct_metrics']['delta']['rmse']:.3f}")
print(f"  MAPE: {report['direct_metrics']['delta']['mape']:.1f}%")
print(f"  Status: Cannot predict delta reliably")

print("\n📉 Theta Parameter:")
print(f"  Status: Using heuristic approach (no ML model)")

print("\n🎯 End-to-End Quality:")
print(f"  Score: {report['end_to_end_metrics']['mean_score']:.3f} (target: >0.7) ❌")
print(f"  Status: Synthetic data quality is poor")

print("\n⚖️  Baseline Comparison:")
ai = report["baseline_comparison"]["ai_score"]
fixed = report["baseline_comparison"]["baseline_fixed"]
heuristic = report["baseline_comparison"]["baseline_heuristic"]
print(f"  AI Model:        {ai:.3f}")
print(f"  Fixed Params:    {fixed:.3f} ({((fixed-ai)/ai*100):+.1f}% better)")
print(f"  Simple Heuristic: {heuristic:.3f} ({((heuristic-ai)/ai*100):+.1f}% better)")
print(f"  Status: ❌ AI is WORSE than baselines")

print("\n💡 Recommended Actions:")
print("  1. CRITICAL: Increase dataset to 25,000 samples")
print("     → Set TESTING_MODE = False in config")
print("  2. Reduce features from 32 to 15")
print("  3. Try simpler models (Ridge, Lasso)")
print("  4. Ensemble AI with heuristic baseline")
print("  5. Improve feature engineering")
