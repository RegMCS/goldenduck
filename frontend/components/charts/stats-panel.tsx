"use client"

import { type DataStatistics } from "@/lib/types"

interface StatsPanelProps {
  stats: DataStatistics
}

function ComparisonRow({
  label,
  historical,
  synthetic,
  isPercent = false,
  lowerIsBetter = false,
  showSign = false,
}: {
  label: string
  historical: number
  synthetic: number
  isPercent?: boolean
  lowerIsBetter?: boolean
  showSign?: boolean
}) {
  const fmt = (v: number) => {
    if (isPercent) {
      return `${showSign && v >= 0 ? "+" : ""}${(v * 100).toFixed(2)}%`
    }
    return `${showSign && v >= 0 ? "+" : ""}${v.toFixed(4)}`
  }

  const absH = Math.abs(historical)
  const absS = Math.abs(synthetic)
  const maxAbs = Math.max(absH, absS, 0.0001)
  const hWidth = (absH / maxAbs) * 100
  const sWidth = (absS / maxAbs) * 100

  const diff = synthetic - historical
  const pctDiff = historical !== 0 ? (diff / Math.abs(historical)) * 100 : 0
  const diffClass =
    Math.abs(pctDiff) < 5
      ? "text-emerald-500"
      : Math.abs(pctDiff) < 20
        ? "text-amber-500"
        : "text-red-500"

  // Colour the bar for each series: neutral grey, highlight better one
  const histIsBetter = lowerIsBetter ? historical < synthetic : historical >= synthetic
  const hBarColor = histIsBetter ? "bg-blue-500" : "bg-blue-400/50"
  const sBarColor = !histIsBetter ? "bg-amber-500" : "bg-amber-400/50"

  return (
    <div className="py-3 border-b border-border/50 last:border-0">
      <div className="flex items-center justify-between mb-2">
        <span className="text-xs font-medium text-muted-foreground">{label}</span>
        <span className={`text-[10px] font-mono font-semibold tabular-nums ${diffClass}`}>
          Δ {pctDiff >= 0 ? "+" : ""}
          {pctDiff.toFixed(1)}%
        </span>
      </div>
      <div className="space-y-1.5">
        <div className="flex items-center gap-2">
          <span className="text-[10px] font-medium text-blue-500 w-10 shrink-0">Hist</span>
          <div className="flex-1 h-1.5 bg-border/60 rounded-full overflow-hidden">
            <div
              className={`h-full rounded-full transition-all duration-700 ${hBarColor}`}
              style={{ width: `${hWidth}%` }}
            />
          </div>
          <span className="text-xs font-mono tabular-nums text-foreground w-20 text-right shrink-0">
            {fmt(historical)}
          </span>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-[10px] font-medium text-amber-500 w-10 shrink-0">Synth</span>
          <div className="flex-1 h-1.5 bg-border/60 rounded-full overflow-hidden">
            <div
              className={`h-full rounded-full transition-all duration-700 ${sBarColor}`}
              style={{ width: `${sWidth}%` }}
            />
          </div>
          <span className="text-xs font-mono tabular-nums text-foreground w-20 text-right shrink-0">
            {fmt(synthetic)}
          </span>
        </div>
      </div>
    </div>
  )
}

function DivergingRow({
  label,
  historical,
  synthetic,
  fmt,
  positiveOnly = false,
}: {
  label: string
  historical: number
  synthetic: number
  fmt: (v: number) => string
  positiveOnly?: boolean
}) {
  const diff = synthetic - historical
  const pctDiff = historical !== 0 ? (diff / Math.abs(historical)) * 100 : 0
  const diffClass =
    Math.abs(pctDiff) < 5
      ? "text-emerald-500"
      : Math.abs(pctDiff) < 20
        ? "text-amber-500"
        : "text-red-500"

  // For diverging bars: find the range across both values
  const maxAbs = positiveOnly
    ? Math.max(historical, synthetic, 0.0001)
    : Math.max(Math.abs(historical), Math.abs(synthetic), 0.0001)

  // Bar positioning: left% is where the bar starts, width% is how wide
  const barProps = (v: number) => {
    if (positiveOnly) {
      return { left: "0%", width: `${(v / maxAbs) * 100}%` }
    }
    const center = 50
    const magnitude = (Math.abs(v) / maxAbs) * 50
    if (v >= 0) return { left: `${center}%`, width: `${magnitude}%` }
    return { left: `${center - magnitude}%`, width: `${magnitude}%` }
  }

  const hBar = barProps(historical)
  const sBar = barProps(synthetic)

  return (
    <div className="py-3 border-b border-border/50 last:border-0">
      <div className="flex items-center justify-between mb-2">
        <span className="text-xs font-medium text-muted-foreground">{label}</span>
        <span className={`text-[10px] font-mono font-semibold tabular-nums ${diffClass}`}>
          Δ {pctDiff >= 0 ? "+" : ""}{pctDiff.toFixed(1)}%
        </span>
      </div>
      <div className="space-y-1.5">
        {/* Historical */}
        <div className="flex items-center gap-2">
          <span className="text-[10px] font-medium text-blue-500 w-10 shrink-0">Hist</span>
          <div className="flex-1 relative h-1.5 bg-border/40 rounded-full overflow-hidden">
            {!positiveOnly && (
              <div className="absolute inset-y-0 left-1/2 w-px bg-border/80" />
            )}
            <div
              className="absolute inset-y-0 bg-blue-500 rounded-full"
              style={{ left: hBar.left, width: hBar.width }}
            />
          </div>
          <span className="text-xs font-mono tabular-nums text-foreground w-20 text-right shrink-0">
            {fmt(historical)}
          </span>
        </div>
        {/* Synthetic */}
        <div className="flex items-center gap-2">
          <span className="text-[10px] font-medium text-amber-500 w-10 shrink-0">Synth</span>
          <div className="flex-1 relative h-1.5 bg-border/40 rounded-full overflow-hidden">
            {!positiveOnly && (
              <div className="absolute inset-y-0 left-1/2 w-px bg-border/80" />
            )}
            <div
              className="absolute inset-y-0 bg-amber-500 rounded-full"
              style={{ left: sBar.left, width: sBar.width }}
            />
          </div>
          <span className="text-xs font-mono tabular-nums text-foreground w-20 text-right shrink-0">
            {fmt(synthetic)}
          </span>
        </div>
        {/* X-axis for diverging rows */}
        {!positiveOnly && (
          <div className="flex items-center gap-2 mt-1">
            <div className="w-10 shrink-0" />
            <div className="flex-1 relative flex items-center justify-between">
              <span className="text-[9px] text-muted-foreground/60">← Negative</span>
              <span className="text-[9px] text-muted-foreground/80 font-mono absolute left-1/2 -translate-x-1/2">0</span>
              <span className="text-[9px] text-muted-foreground/60">Positive →</span>
            </div>
            <div className="w-20 shrink-0" />
          </div>
        )}
      </div>
    </div>
  )
}

export function StatsPanel({ stats }: StatsPanelProps) {
  const { historical: h, synthetic: s } = stats
  const hVar95 = typeof h.var95 === "number" ? h.var95 : 0
  const sVar95 = typeof s.var95 === "number" ? s.var95 : 0
  const hLegacyMomentum = typeof h.acfLag1 === "number" ? Math.max(0, Math.min(1, 0.5 + 0.5 * h.acfLag1)) : 0.5
  const sLegacyMomentum = typeof s.acfLag1 === "number" ? Math.max(0, Math.min(1, 0.5 + 0.5 * s.acfLag1)) : 0.5
  const hHurstMomentum = typeof h.hurstMomentum === "number" ? h.hurstMomentum : hLegacyMomentum
  const sHurstMomentum = typeof s.hurstMomentum === "number" ? s.hurstMomentum : sLegacyMomentum

  return (
    <div className="grid gap-4 md:grid-cols-2">
      {/* Performance metrics */}
      <div className="rounded-xl border border-border bg-card p-5">
        <div className="flex items-center gap-2 mb-4">
          <div className="h-3 w-3 rounded-full bg-gradient-to-br from-blue-500 to-amber-500" />
          <h4 className="text-sm font-semibold text-foreground">Performance</h4>
        </div>
        <ComparisonRow
          label="Annualised Return"
          historical={h.annualizedReturn}
          synthetic={s.annualizedReturn}
          isPercent
        />
        <ComparisonRow
          label="Annualised Volatility"
          historical={h.annualizedVol}
          synthetic={s.annualizedVol}
          isPercent
          lowerIsBetter
        />
        <ComparisonRow
          label="Sharpe Ratio"
          historical={h.sharpe}
          synthetic={s.sharpe}
        />
        <ComparisonRow
          label="Max Drawdown"
          historical={Math.abs(h.maxDrawdown)}
          synthetic={Math.abs(s.maxDrawdown)}
          isPercent
          lowerIsBetter
        />
        <ComparisonRow
          label="Total Return"
          historical={h.totalReturn}
          synthetic={s.totalReturn}
          isPercent
        />
      </div>

      {/* Distribution properties */}
      <div className="rounded-xl border border-border bg-card p-5">
        <div className="flex items-center gap-2 mb-4">
          <div className="h-3 w-3 rounded-full bg-gradient-to-br from-violet-500 to-pink-500" />
          <h4 className="text-sm font-semibold text-foreground">Distribution</h4>
        </div>
        <DivergingRow label="Daily Mean Return [Trend]" historical={h.mean} synthetic={s.mean} fmt={(v) => v.toFixed(4)} />
        <DivergingRow label="Daily Std Dev [Volatility]" historical={h.std} synthetic={s.std} fmt={(v) => v.toFixed(4)} positiveOnly />
        <DivergingRow label="Skewness" historical={h.skewness} synthetic={s.skewness} fmt={(v) => `${v >= 0 ? "+" : ""}${v.toFixed(4)}`} />
        <DivergingRow label="Excess Kurtosis [Fat Tails]" historical={h.kurtosis} synthetic={s.kurtosis} fmt={(v) => `${v >= 0 ? "+" : ""}${v.toFixed(4)}`} />
        <DivergingRow
          label="Hurst Momentum [Momentum]"
          historical={hHurstMomentum}
          synthetic={sHurstMomentum}
          fmt={(v) => v.toFixed(4)}
          positiveOnly
        />
        <DivergingRow
          label="Daily VaR (95%)"
          historical={hVar95}
          synthetic={sVar95}
          fmt={(v) => `${(v * 100).toFixed(2)}%`}
          positiveOnly
        />
          
        <div className="py-3">
          <div className="flex items-center justify-between mb-1">
            <span className="text-xs font-medium text-muted-foreground">Data Points</span>
          </div>
          <div className="flex justify-between text-xs font-mono">
            <span className="text-blue-500">{h.numDataPoints.toLocaleString()}</span>
            <span className="text-muted-foreground">vs</span>
            <span className="text-amber-500">{s.numDataPoints.toLocaleString()}</span>
          </div>
        </div>
      </div>
    </div>
  )
}
