"use client"

import { useMemo, useCallback } from "react"
import {
  Area,
  AreaChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts"
import { type ReturnPoint } from "@/lib/types"

interface DistributionOverlayChartProps {
  data: ReturnPoint[]
  historicalKurtosis?: number
  syntheticKurtosis?: number
  height?: number
}

function buildHistogram(values: number[], minV: number, maxV: number, bins = 40): number[] {
  if (values.length === 0 || maxV <= minV) return new Array(bins).fill(0)
  const counts = new Array(bins).fill(0)
  const width = (maxV - minV) / bins

  for (const v of values) {
    if (!Number.isFinite(v)) continue
    const idx = v <= minV ? 0 : v >= maxV ? bins - 1 : Math.min(bins - 1, Math.max(0, Math.floor((v - minV) / width)))
    counts[idx] += 1
  }

  // Probability density approximation so area ~ 1
  return counts.map((c) => c / (values.length * width))
}

function excessKurtosis(values: number[]): number {
  if (values.length < 4) return 0
  const n = values.length
  const mean = values.reduce((a, b) => a + b, 0) / n
  const centered = values.map((v) => v - mean)
  const m2 = centered.reduce((a, b) => a + b * b, 0) / n
  if (m2 <= 1e-12) return 0
  const m4 = centered.reduce((a, b) => a + b * b * b * b, 0) / n
  return m4 / (m2 * m2) - 3
}

export function DistributionOverlayChart({
  data,
  historicalKurtosis,
  syntheticKurtosis,
  height = 420,
}: DistributionOverlayChartProps) {
  const { chartData, histKurtComputed, synthKurtComputed } = useMemo(() => {
    const h = data.map((d) => d.historicalReturn * 100).filter((v) => Number.isFinite(v))
    const s = data.map((d) => d.syntheticReturn * 100).filter((v) => Number.isFinite(v))
    const all = [...h, ...s]

    if (all.length < 20) {
      return {
        chartData: [] as Array<{ x: number; historicalDensity: number; syntheticDensity: number; left: number; right: number }>,
        histKurtComputed: 0,
        synthKurtComputed: 0,
      }
    }

    const sorted = [...all].sort((a, b) => a - b)
    const q01 = sorted[Math.floor(0.01 * (sorted.length - 1))]
    const q99 = sorted[Math.floor(0.99 * (sorted.length - 1))]
    const minV = Number.isFinite(q01) ? q01 : sorted[0]
    const maxV = Number.isFinite(q99) ? q99 : sorted[sorted.length - 1]
    const bins = 40
    const width = (maxV - minV) / bins

    const hDen = buildHistogram(h, minV, maxV, bins)
    const sDen = buildHistogram(s, minV, maxV, bins)

    const rows = new Array(bins).fill(0).map((_, i) => {
      const left = minV + i * width
      const right = left + width
      const center = (left + right) / 2
      return {
        x: center,
        left,
        right,
        historicalDensity: hDen[i],
        syntheticDensity: sDen[i],
      }
    })

    return {
      chartData: rows,
      histKurtComputed: excessKurtosis(h),
      synthKurtComputed: excessKurtosis(s),
    }
  }, [data])

  const histKurt =
    typeof historicalKurtosis === "number" && Number.isFinite(historicalKurtosis)
      ? historicalKurtosis
      : histKurtComputed
  const synthKurt =
    typeof syntheticKurtosis === "number" && Number.isFinite(syntheticKurtosis)
      ? syntheticKurtosis
      : synthKurtComputed

  const CustomTooltip = useCallback(
    ({ active, payload, label }: { active?: boolean; payload?: Array<{ name: string; value: number; payload: { left: number; right: number } }>; label?: number }) => {
      if (!active || !payload || payload.length === 0) return null
      const p0 = payload[0]
      const left = p0?.payload?.left ?? label ?? 0
      const right = p0?.payload?.right ?? label ?? 0
      const hVal = payload.find((p) => p.name === "historicalDensity")?.value ?? 0
      const sVal = payload.find((p) => p.name === "syntheticDensity")?.value ?? 0

      return (
        <div className="rounded-lg border border-border bg-card/95 p-3 shadow-xl backdrop-blur">
          <p className="mb-2 text-xs font-mono text-muted-foreground">
            Return bin: [{left.toFixed(2)}%, {right.toFixed(2)}%]
          </p>
          <div className="space-y-1.5 text-xs">
            <div className="flex items-center justify-between gap-8">
              <span className="text-muted-foreground">Historical density</span>
              <span className="font-mono font-semibold text-blue-500">{hVal.toFixed(4)}</span>
            </div>
            <div className="flex items-center justify-between gap-8">
              <span className="text-muted-foreground">Synthetic density</span>
              <span className="font-mono font-semibold text-amber-500">{sVal.toFixed(4)}</span>
            </div>
          </div>
        </div>
      )
    },
    []
  )

  if (chartData.length === 0) {
    return <p className="text-sm text-muted-foreground">Distribution data not available</p>
  }

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between gap-4">
        <div>
          <h4 className="text-sm font-semibold text-foreground">Return Distribution Overlay</h4>
          <p className="text-xs text-muted-foreground mt-0.5">Histogram-density overlay (Hist vs Synth)</p>
        </div>
        <div className="flex items-center gap-3 text-xs">
          <span className="text-muted-foreground">Hist kurt:</span>
          <span className="font-mono font-semibold text-blue-500">{histKurt.toFixed(3)}</span>
          <span className="text-muted-foreground">Synth kurt:</span>
          <span className="font-mono font-semibold text-amber-500">{synthKurt.toFixed(3)}</span>
        </div>
      </div>

      <ResponsiveContainer width="100%" height={height}>
        <AreaChart data={chartData} margin={{ top: 8, right: 12, bottom: 10, left: 10 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" opacity={0.4} />
          <XAxis
            dataKey="x"
            tick={{ fontSize: 10, fill: "var(--muted-foreground)" }}
            tickLine={false}
            axisLine={{ stroke: "var(--border)" }}
            tickFormatter={(v: number) => `${v.toFixed(1)}%`}
          />
          <YAxis
            tick={{ fontSize: 10, fill: "var(--muted-foreground)" }}
            tickLine={false}
            axisLine={{ stroke: "var(--border)" }}
            width={52}
          />
          <Tooltip content={<CustomTooltip />} />
          <Area
            type="monotone"
            dataKey="historicalDensity"
            stroke="#3b82f6"
            strokeWidth={2}
            fill="#3b82f6"
            fillOpacity={0.18}
            dot={false}
            name="historicalDensity"
          />
          <Area
            type="monotone"
            dataKey="syntheticDensity"
            stroke="#f59e0b"
            strokeWidth={2}
            fill="#f59e0b"
            fillOpacity={0.15}
            dot={false}
            name="syntheticDensity"
            strokeDasharray="5 2"
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  )
}
