"use client"

import { useMemo, useCallback } from "react"
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
  ReferenceLine,
} from "recharts"
import { type ReturnPoint } from "@/lib/types"

interface CumulativeReturnChartProps {
  data: ReturnPoint[]
  height?: number
}

export function CumulativeReturnChart({
  data,
  height = 400,
}: CumulativeReturnChartProps) {
  const displayData = useMemo(() => {
    const maxPoints = 300
    const step = Math.max(1, Math.floor(data.length / maxPoints))
    return data.filter((_, i) => i % step === 0).map((d) => ({
      ...d,
      historicalCumPct: d.historicalCumReturn * 100,
      syntheticCumPct: d.syntheticCumReturn * 100,
    }))
  }, [data])

  const { histFinalReturn, synthFinalReturn, trackingError } = useMemo(() => {
    if (data.length === 0) return { histFinalReturn: 0, synthFinalReturn: 0, trackingError: 0 }
    const last = data[data.length - 1]
    const diffs = data.map((d) => d.historicalCumReturn - d.syntheticCumReturn)
    const meanDiff = diffs.reduce((a, b) => a + b, 0) / diffs.length
    const teVariance = diffs.reduce((a, b) => a + (b - meanDiff) ** 2, 0) / (diffs.length - 1)
    return {
      histFinalReturn: last.historicalCumReturn * 100,
      synthFinalReturn: last.syntheticCumReturn * 100,
      trackingError: Math.sqrt(teVariance * 252) * 100,
    }
  }, [data])

  const formatDate = useCallback((value: string) => {
    if (!value) return ""
    const parts = value.split("-")
    return `${parts[1]}/${parts[2]}`
  }, [])

  const CustomTooltip = useCallback(
    ({ active, payload, label }: { active?: boolean; payload?: Array<{ value: number; name: string }>; label?: string }) => {
      if (!active || !payload?.length) return null
      return (
        <div className="rounded-lg border border-border bg-card p-3 shadow-lg">
          <p className="mb-2 text-xs font-mono text-muted-foreground">{label}</p>
          <div className="space-y-1">
            {payload.map((entry) => {
              const isHist = entry.name === "historicalCumPct"
              return (
                <div key={entry.name} className="flex items-center justify-between gap-6 text-xs">
                  <span className="flex items-center gap-1.5">
                    <span
                      className="inline-block h-2 w-2 rounded-full"
                      style={{ backgroundColor: isHist ? "#3b82f6" : "#f59e0b" }}
                    />
                    <span className="text-muted-foreground">
                      {isHist ? "Historical" : "Synthetic"}
                    </span>
                  </span>
                  <span
                    className={`font-mono font-medium ${entry.value >= 0 ? "text-emerald-500" : "text-red-500"}`}
                  >
                    {entry.value >= 0 ? "+" : ""}{entry.value.toFixed(2)}%
                  </span>
                </div>
              )
            })}
          </div>
        </div>
      )
    },
    []
  )

  return (
    <div className="space-y-3">
      <div className="flex items-baseline justify-between">
        <div>
          <h4 className="text-sm font-semibold text-foreground">Cumulative Returns</h4>
          <p className="text-xs text-muted-foreground">
            Log-return compounded over time
          </p>
        </div>
        <div className="flex items-center gap-4 text-xs">
          <div className="flex items-center gap-1">
            <span className="inline-block h-2 w-2 rounded-full bg-blue-500" />
            <span className="text-muted-foreground">Hist:</span>
            <span
              className={`font-mono font-medium ${histFinalReturn >= 0 ? "text-emerald-500" : "text-red-500"}`}
            >
              {histFinalReturn >= 0 ? "+" : ""}{histFinalReturn.toFixed(2)}%
            </span>
          </div>
          <div className="flex items-center gap-1">
            <span className="inline-block h-2 w-2 rounded-full bg-amber-500" />
            <span className="text-muted-foreground">Synth:</span>
            <span
              className={`font-mono font-medium ${synthFinalReturn >= 0 ? "text-emerald-500" : "text-red-500"}`}
            >
              {synthFinalReturn >= 0 ? "+" : ""}{synthFinalReturn.toFixed(2)}%
            </span>
          </div>
          <div className="flex items-center gap-1">
            <span className="text-muted-foreground">TE (ann.):</span>
            <span className="font-mono font-medium text-foreground">
              {trackingError.toFixed(2)}%
            </span>
          </div>
        </div>
      </div>
      <ResponsiveContainer width="100%" height={height}>
        <AreaChart data={displayData} margin={{ top: 10, right: 10, bottom: 0, left: 10 }}>
          <defs>
            <linearGradient id="histGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.15} />
              <stop offset="95%" stopColor="#3b82f6" stopOpacity={0} />
            </linearGradient>
            <linearGradient id="synthGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#f59e0b" stopOpacity={0.15} />
              <stop offset="95%" stopColor="#f59e0b" stopOpacity={0} />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" opacity={0.5} />
          <XAxis
            dataKey="date"
            tickFormatter={formatDate}
            tick={{ fontSize: 10, fill: "var(--muted-foreground)" }}
            tickLine={false}
            axisLine={{ stroke: "var(--border)" }}
            interval={Math.max(1, Math.floor(displayData.length / 8))}
          />
          <YAxis
            tick={{ fontSize: 10, fill: "var(--muted-foreground)" }}
            tickLine={false}
            axisLine={{ stroke: "var(--border)" }}
            tickFormatter={(v: number) => `${v.toFixed(0)}%`}
            width={55}
          />
          <Tooltip content={<CustomTooltip />} />
          <Legend
            wrapperStyle={{ fontSize: 11 }}
            formatter={(value: string) => (
              <span className="text-xs text-muted-foreground">
                {value === "historicalCumPct" ? "Historical" : "Synthetic"}
              </span>
            )}
          />
          <ReferenceLine y={0} stroke="var(--muted-foreground)" strokeDasharray="3 3" opacity={0.5} />
          <Area
            type="monotone"
            dataKey="historicalCumPct"
            stroke="#3b82f6"
            fill="url(#histGrad)"
            strokeWidth={1.5}
            dot={false}
            name="historicalCumPct"
          />
          <Area
            type="monotone"
            dataKey="syntheticCumPct"
            stroke="#f59e0b"
            fill="url(#synthGrad)"
            strokeWidth={1.5}
            dot={false}
            name="syntheticCumPct"
            strokeDasharray="4 2"
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  )
}
