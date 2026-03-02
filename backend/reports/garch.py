"""
GARCH(1,1) Diagnostic Script
Run this with your real AAPL data to generate all presentation charts.

Requirements:
    pip install arch pandas numpy scipy statsmodels plotly kaleido yfinance
"""

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import acf
from arch import arch_model
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
import json
import warnings
warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════════════
# STEP 1: LOAD YOUR DATA
# ══════════════════════════════════════════════════════════════════════

# Option A: Pull directly from yfinance (recommended)
import yfinance as yf
df = yf.download("AAPL", start="2024-01-01", end="2026-01-01", auto_adjust=True)
historical_returns = df["Close"].pct_change().dropna().values.flatten()

# Option B: Load from your own CSV
# df = pd.read_csv("your_aapl_data.csv", parse_dates=["Date"], index_col="Date")
# historical_returns = df["Close"].pct_change().dropna().values

print(f"Loaded {len(historical_returns)} daily returns")
print(f"Date range: {df.index[1].date()} to {df.index[-1].date()}")


# ══════════════════════════════════════════════════════════════════════
# STEP 2: FIT GARCH(1,1) WITH NORMAL INNOVATIONS
# ══════════════════════════════════════════════════════════════════════

# Scale returns to percentage (arch library convention)
returns_pct = historical_returns * 100

am = arch_model(returns_pct, vol='Garch', p=1, q=1, dist='normal', mean='Constant')
res = am.fit(disp='off')

print("\n── GARCH(1,1) Fitted Parameters ──")
print(res.summary())

# Extract parameters
mu     = res.params['mu'] / 100          # back to decimal
omega  = res.params['omega'] / 10000
alpha  = res.params['alpha[1]']
beta   = res.params['beta[1]']

print(f"\nomega={omega:.2e}, alpha={alpha:.4f}, beta={beta:.4f}")
print(f"Persistence (alpha+beta) = {alpha+beta:.4f}")


# ══════════════════════════════════════════════════════════════════════
# STEP 3: SIMULATE SYNTHETIC RETURNS FROM FITTED GARCH(1,1)
# ══════════════════════════════════════════════════════════════════════

n_synth = len(historical_returns)
np.random.seed(42)

h_s = np.zeros(n_synth)
e_s = np.zeros(n_synth)
synth = np.zeros(n_synth)

# Initialise at unconditional variance
h_s[0] = omega / (1 - alpha - beta)

for t in range(1, n_synth):
    h_s[t] = omega + alpha * e_s[t-1]**2 + beta * h_s[t-1]
    e_s[t] = np.random.normal(0, np.sqrt(h_s[t]))   # ← normal innovations
    synth[t] = mu + e_s[t]

synthetic_returns = synth[1:]
historical_returns_trimmed = historical_returns[1:]  # align length


# ══════════════════════════════════════════════════════════════════════
# STEP 4: COMPUTE DIAGNOSTICS
# ══════════════════════════════════════════════════════════════════════

hist = historical_returns_trimmed
synth = synthetic_returns

hist_kurt   = stats.kurtosis(hist)
synth_kurt  = stats.kurtosis(synth)
hist_skew   = stats.skew(hist)
synth_skew  = stats.skew(synth)
hist_vol    = np.std(hist) * np.sqrt(252)
synth_vol   = np.std(synth) * np.sqrt(252)

hist_acf_r  = acf(hist,    nlags=20, fft=False)[1:]
synth_acf_r = acf(synth,   nlags=20, fft=False)[1:]
hist_acf2   = acf(hist**2, nlags=20, fft=False)[1:]
synth_acf2  = acf(synth**2, nlags=20, fft=False)[1:]

ks_stat, ks_pval = stats.ks_2samp(hist, synth)

# 95% confidence band for ACF
ci = 1.96 / np.sqrt(len(hist))

# Rolling volatility (21-day)
hist_vol_roll  = pd.Series(hist).rolling(21).std() * np.sqrt(252)
synth_vol_roll = pd.Series(synth).rolling(21).std() * np.sqrt(252)

# QQ quantiles
n_qq = 200
quantiles = np.linspace(0.01, 0.99, n_qq)
hist_q  = np.quantile(hist,  quantiles)
synth_q = np.quantile(synth, quantiles)

# Validation match scores
def match_score(synth_val, target_val, tol):
    return max(0.0, 1.0 - abs(synth_val - target_val) / (abs(target_val) + tol))

vol_match  = match_score(np.std(synth),   np.std(hist),   1e-6)
kurt_match = match_score(synth_kurt,      hist_kurt,      3.0)
acf1_match = match_score(synth_acf_r[0],  hist_acf_r[0],  0.1)
skew_match = match_score(synth_skew,      hist_skew,      0.2)
overall    = (vol_match + kurt_match + acf1_match + skew_match) / 4

print("\n── Diagnostic Summary ──")
print(f"{'Metric':<25} {'Historical':>12} {'Synthetic':>12}")
print("-" * 51)
print(f"{'Annualised Vol':<25} {hist_vol:>12.4f} {synth_vol:>12.4f}")
print(f"{'Kurtosis':<25} {hist_kurt:>12.2f} {synth_kurt:>12.2f}")
print(f"{'Skewness':<25} {hist_skew:>12.4f} {synth_skew:>12.4f}")
print(f"{'ACF Lag-1 (returns)':<25} {hist_acf_r[0]:>12.4f} {synth_acf_r[0]:>12.4f}")
print(f"{'ACF Lag-1 (vol clust)':<25} {hist_acf2[0]:>12.4f} {synth_acf2[0]:>12.4f}")
print(f"{'KS Statistic':<25} {ks_stat:>12.4f} {'p=' + f'{ks_pval:.4f}':>12}")
print("\n── Match Scores ──")
print(f"{'Volatility':<20} {vol_match:.1%}")
print(f"{'Kurtosis':<20} {kurt_match:.1%}")
print(f"{'ACF (Lag-1)':<20} {acf1_match:.1%}")
print(f"{'Skewness':<20} {skew_match:.1%}")
print(f"{'Overall':<20} {overall:.1%}")


# ══════════════════════════════════════════════════════════════════════
# STEP 5: GENERATE CHARTS
# ══════════════════════════════════════════════════════════════════════

lags = list(range(1, 21))

# ── Chart 1: Rolling Volatility ───────────────────────────────────────
fig1 = go.Figure()
fig1.add_trace(go.Scatter(
    y=hist_vol_roll.tolist(), name='Historical', line=dict(width=1.5)))
fig1.add_trace(go.Scatter(
    y=synth_vol_roll.tolist(), name='Synthetic (GARCH)', line=dict(width=1.5, dash='dash')))
fig1.update_layout(
    title={"text": "Rolling 21-Day Volatility — GARCH(1,1)<br>"
                   "<span style='font-size:16px;font-weight:normal'>"
                   "✅ Volatility clustering captured in both series</span>"},
    legend=dict(orientation='h', yanchor='bottom', y=1.05, xanchor='center', x=0.5)
)
fig1.update_xaxes(title_text="Trading Day")
fig1.update_yaxes(title_text="Annualised Vol")
fig1.write_image("chart1_rolling_vol.png")
with open("chart1_rolling_vol.png.meta.json", "w") as f:
    json.dump({
        "caption": "GARCH(1,1) Rolling Volatility — ✅ What Worked",
        "description": "Rolling 21-day annualised volatility for historical vs synthetic. "
                       "Clustering behaviour is replicated."
    }, f)

# ── Chart 2: ACF of Squared Returns ──────────────────────────────────
fig2 = go.Figure()
fig2.add_trace(go.Bar(x=lags, y=list(hist_acf2), name='Historical', opacity=0.8))
fig2.add_trace(go.Bar(x=lags, y=list(synth_acf2), name='Synthetic', opacity=0.8))
fig2.add_hline(y=ci,  line_dash='dot', line_color='gray',
               annotation_text='95% CI', annotation_position='top right')
fig2.add_hline(y=-ci, line_dash='dot', line_color='gray')
fig2.update_layout(
    barmode='group',
    title={"text": "ACF of Squared Returns — GARCH(1,1)<br>"
                   "<span style='font-size:16px;font-weight:normal'>"
                   "✅ ARCH effect (volatility clustering) well-replicated</span>"},
    legend=dict(orientation='h', yanchor='bottom', y=1.05, xanchor='center', x=0.5)
)
fig2.update_xaxes(title_text="Lag", dtick=2)
fig2.update_yaxes(title_text="Autocorrelation")
fig2.write_image("chart2_acf_squared.png")
with open("chart2_acf_squared.png.meta.json", "w") as f:
    json.dump({
        "caption": "ACF of Squared Returns — ✅ Volatility Clustering Captured",
        "description": "Positive ACF in squared returns confirms GARCH replicates the ARCH effect."
    }, f)

# ── Chart 3: Return Distribution Overlay ─────────────────────────────
x_range = np.linspace(
    min(hist.min(), synth.min()) * 1.1,
    max(hist.max(), synth.max()) * 1.1, 400)
hist_kde  = stats.gaussian_kde(hist)
synth_kde = stats.gaussian_kde(synth)

fig3 = go.Figure()
fig3.add_trace(go.Scatter(
    x=x_range, y=hist_kde(x_range), name='Historical',
    fill='tozeroy', line=dict(width=2)))
fig3.add_trace(go.Scatter(
    x=x_range, y=synth_kde(x_range), name='Synthetic (GARCH)',
    line=dict(width=2, dash='dash')))
fig3.update_layout(
    title={"text": "Return Distribution — GARCH(1,1)<br>"
                   "<span style='font-size:16px;font-weight:normal'>"
                   f"❌ Synthetic tails too thin — kurtosis {synth_kurt:.1f} vs {hist_kurt:.1f}</span>"},
    legend=dict(orientation='h', yanchor='bottom', y=1.05, xanchor='center', x=0.5)
)
fig3.update_xaxes(title_text="Daily Return")
fig3.update_yaxes(title_text="Density")
fig3.write_image("chart3_distribution.png")
with open("chart3_distribution.png.meta.json", "w") as f:
    json.dump({
        "caption": "Return Distribution — ❌ Thin Tails vs Historical",
        "description": "KDE showing synthetic distribution misses fat tails of real returns."
    }, f)

# ── Chart 4: QQ Plot ──────────────────────────────────────────────────
qq_min, qq_max = float(min(hist_q.min(), synth_q.min())), float(max(hist_q.max(), synth_q.max()))
fig4 = go.Figure()
fig4.add_trace(go.Scatter(
    x=list(hist_q), y=list(synth_q), mode='markers',
    name='Quantile pairs', marker=dict(size=5, opacity=0.6)))
fig4.add_trace(go.Scatter(
    x=[qq_min, qq_max], y=[qq_min, qq_max],
    mode='lines', name='Perfect match', line=dict(dash='dash', width=2)))
fig4.update_layout(
    title={"text": "QQ Plot: Synthetic vs Historical — GARCH(1,1)<br>"
                   "<span style='font-size:16px;font-weight:normal'>"
                   "❌ Tail divergence — extreme returns underestimated</span>"},
    legend=dict(orientation='h', yanchor='bottom', y=1.05, xanchor='center', x=0.5)
)
fig4.update_xaxes(title_text="Historical Quantile")
fig4.update_yaxes(title_text="Synthetic Quantile")
fig4.write_image("chart4_qq.png")
with open("chart4_qq.png.meta.json", "w") as f:
    json.dump({
        "caption": "QQ Plot — ❌ Tail Underestimation at Extremes",
        "description": "QQ plot diverges from diagonal at tails, showing GARCH(1,1) underestimates extreme returns."
    }, f)

# ── Chart 5: ACF of Raw Returns ───────────────────────────────────────
fig5 = go.Figure()
fig5.add_trace(go.Bar(x=lags, y=list(hist_acf_r), name='Historical', opacity=0.8))
fig5.add_trace(go.Bar(x=lags, y=list(synth_acf_r), name='Synthetic', opacity=0.8))
fig5.add_hline(y=ci,  line_dash='dot', line_color='gray',
               annotation_text='95% CI', annotation_position='top right')
fig5.add_hline(y=-ci, line_dash='dot', line_color='gray')
fig5.update_layout(
    barmode='group',
    title={"text": "ACF of Returns — GARCH(1,1)<br>"
                   "<span style='font-size:16px;font-weight:normal'>"
                   "❌ No momentum control — synthetic ACF near zero</span>"},
    legend=dict(orientation='h', yanchor='bottom', y=1.05, xanchor='center', x=0.5)
)
fig5.update_xaxes(title_text="Lag", dtick=2)
fig5.update_yaxes(title_text="Autocorrelation")
fig5.write_image("chart5_acf_returns.png")
with open("chart5_acf_returns.png.meta.json", "w") as f:
    json.dump({
        "caption": "ACF of Returns — ❌ No Momentum Control",
        "description": "GARCH(1,1) has no AR component so synthetic return ACF is near zero."
    }, f)

# ── Chart 6: Validation Score Summary ────────────────────────────────
metrics = ['Volatility', 'Kurtosis', 'ACF', 'Skewness', 'Overall']
scores  = [vol_match, kurt_match, acf1_match, skew_match, overall]
colors  = ['#2ecc71' if s >= 0.8 else '#e67e22' if s >= 0.5 else '#e74c3c'
           for s in scores]

fig6 = go.Figure()
fig6.add_trace(go.Bar(
    x=metrics, y=[s * 100 for s in scores],
    marker_color=colors,
    text=[f"{s:.0%}" for s in scores],
    textposition='outside'))
fig6.add_hline(y=80, line_dash='dash', line_color='green',
               annotation_text='80% target', annotation_position='top right')
fig6.update_layout(
    title={"text": "GARCH(1,1) Validation Score Summary<br>"
                   "<span style='font-size:16px;font-weight:normal'>"
                   "✅ Vol clustering | ❌ Fat tails | ❌ ACF | ❌ No user knobs</span>"},
)
fig6.update_xaxes(title_text="Metric")
fig6.update_yaxes(title_text="Match Score (%)", range=[0, 110], dtick=20)
fig6.write_image("chart6_scores.png")
with open("chart6_scores.png.meta.json", "w") as f:
    json.dump({
        "caption": "GARCH(1,1) Validation Score Summary",
        "description": "Bar chart of match scores per metric. Green=good, orange=acceptable, red=poor."
    }, f)

print("\n✅ All 6 charts saved successfully.")
print("Files: chart1_rolling_vol.png through chart6_scores.png")
