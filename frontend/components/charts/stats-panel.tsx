"use client"

import { type DataStatistics } from "@/lib/types"

interface StatsPanelProps {
  stats: DataStatistics
}

function formatStat(value: number, isPercent = false): string {
  if (isPercent) {
    return `${(value * 100).toFixed(2)}%`
  }
  return value.toFixed(4)
}

function StatRow({
  label,
  historical,
  synthetic,
  isPercent = false,
  highlightDiff = false,
}: {
  label: string
  historical: number
  synthetic: number
  isPercent?: boolean
  highlightDiff?: boolean
}) {
  const diff = synthetic - historical
  const pctDiff = historical !== 0 ? ((diff / Math.abs(historical)) * 100) : 0

  return (
    <tr className="border-b border-border/50 last:border-0">
      <td className="py-2 pr-4 text-xs text-muted-foreground whitespace-nowrap">{label}</td>
      <td className="py-2 px-2 text-xs font-mono text-foreground text-right tabular-nums">
        {formatStat(historical, isPercent)}
      </td>
      <td className="py-2 px-2 text-xs font-mono text-foreground text-right tabular-nums">
        {formatStat(synthetic, isPercent)}
      </td>
      {highlightDiff && (
        <td className="py-2 pl-2 text-xs font-mono text-right tabular-nums">
          <span
            className={
              Math.abs(pctDiff) < 5
                ? "text-emerald-500"
                : Math.abs(pctDiff) < 15
                  ? "text-amber-500"
                  : "text-red-500"
            }
          >
            {pctDiff >= 0 ? "+" : ""}{pctDiff.toFixed(1)}%
          </span>
        </td>
      )}
    </tr>
  )
}

export function StatsPanel({ stats }: StatsPanelProps) {
  const { historical: h, synthetic: s } = stats

  return (
    <div className="space-y-4">
      <div className="rounded-xl border border-border bg-card p-4">
        <h4 className="mb-3 text-sm font-semibold text-foreground">
          Summary Statistics
        </h4>
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="border-b border-border">
                <th className="pb-2 pr-4 text-left text-xs font-medium text-muted-foreground">Metric</th>
                <th className="pb-2 px-2 text-right text-xs font-medium text-blue-500">Historical</th>
                <th className="pb-2 px-2 text-right text-xs font-medium text-amber-500">Synthetic</th>
                <th className="pb-2 pl-2 text-right text-xs font-medium text-muted-foreground">Diff</th>
              </tr>
            </thead>
            <tbody>
              <StatRow label="Ann. Return" historical={h.annualizedReturn} synthetic={s.annualizedReturn} isPercent highlightDiff />
              <StatRow label="Ann. Vol" historical={h.annualizedVol} synthetic={s.annualizedVol} isPercent highlightDiff />
              <StatRow label="Sharpe Ratio" historical={h.sharpe} synthetic={s.sharpe} highlightDiff />
              <StatRow label="Max Drawdown" historical={h.maxDrawdown} synthetic={s.maxDrawdown} isPercent highlightDiff />
              <StatRow label="Total Return" historical={h.totalReturn} synthetic={s.totalReturn} isPercent highlightDiff />
            </tbody>
          </table>
        </div>
      </div>

      <div className="rounded-xl border border-border bg-card p-4">
        <h4 className="mb-3 text-sm font-semibold text-foreground">
          Distribution Properties
        </h4>
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="border-b border-border">
                <th className="pb-2 pr-4 text-left text-xs font-medium text-muted-foreground">Metric</th>
                <th className="pb-2 px-2 text-right text-xs font-medium text-blue-500">Historical</th>
                <th className="pb-2 px-2 text-right text-xs font-medium text-amber-500">Synthetic</th>
                <th className="pb-2 pl-2 text-right text-xs font-medium text-muted-foreground">Diff</th>
              </tr>
            </thead>
            <tbody>
              <StatRow label="Daily Mean" historical={h.mean} synthetic={s.mean} highlightDiff />
              <StatRow label="Daily Std" historical={h.std} synthetic={s.std} highlightDiff />
              <StatRow label="Skewness" historical={h.skewness} synthetic={s.skewness} highlightDiff />
              <StatRow label="Excess Kurtosis" historical={h.kurtosis} synthetic={s.kurtosis} highlightDiff />
              <StatRow label="Data Points" historical={h.numDataPoints} synthetic={s.numDataPoints} />
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}
