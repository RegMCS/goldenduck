"use client"

import { useMemo, useCallback } from "react"
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from "recharts"
import { type TimeSeriesPoint } from "@/lib/types"

interface PriceOverlayChartProps {
  data: TimeSeriesPoint[]
  height?: number
}

export function PriceOverlayChart({ data, height = 400 }: PriceOverlayChartProps) {
  const displayData = useMemo(() => {
    const maxPoints = 300
    const step = Math.max(1, Math.floor(data.length / maxPoints))
    return data.filter((_, i) => i % step === 0)
  }, [data])

  const correlation = useMemo(() => {
    if (data.length < 2) return 0
    const hMean = data.reduce((s, d) => s + d.historical, 0) / data.length
    const sMean = data.reduce((s, d) => s + d.synthetic, 0) / data.length
    let cov = 0, hVar = 0, sVar = 0
    for (const d of data) {
      const hd = d.historical - hMean
      const sd = d.synthetic - sMean
      cov += hd * sd
      hVar += hd * hd
      sVar += sd * sd
    }
    return cov / Math.sqrt(hVar * sVar)
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
            {payload.map((entry) => (
              <div key={entry.name} className="flex items-center justify-between gap-6 text-xs">
                <span className="flex items-center gap-1.5">
                  <span
                    className="inline-block h-2 w-2 rounded-full"
                    style={{
                      backgroundColor: entry.name === "historical" ? "#3b82f6" : "#f59e0b",
                    }}
                  />
                  <span className="text-muted-foreground capitalize">{entry.name}</span>
                </span>
                <span className="font-mono font-medium text-foreground">
                  {entry.value.toFixed(2)}
                </span>
              </div>
            ))}
            {payload.length === 2 && (
              <div className="flex items-center justify-between gap-6 text-xs border-t border-border pt-1 mt-1">
                <span className="text-muted-foreground">Spread</span>
                <span className="font-mono font-medium text-foreground">
                  {(payload[0].value - payload[1].value).toFixed(2)}
                </span>
              </div>
            )}
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
          <h4 className="text-sm font-semibold text-foreground">Price Time Series Overlay</h4>
          <p className="text-xs text-muted-foreground">
            Historical vs synthetic close prices
          </p>
        </div>
        <div className="flex items-center gap-4 text-xs">
          <div className="flex items-center gap-1">
            <span className="text-muted-foreground">Pearson r:</span>
            <span className="font-mono font-medium text-foreground">
              {correlation.toFixed(4)}
            </span>
          </div>
          <div className="flex items-center gap-1">
            <span className="text-muted-foreground">N:</span>
            <span className="font-mono font-medium text-foreground">
              {data.length.toLocaleString()}
            </span>
          </div>
        </div>
      </div>
      <ResponsiveContainer width="100%" height={height}>
        <LineChart data={displayData} margin={{ top: 10, right: 10, bottom: 0, left: 10 }}>
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
            tickFormatter={(v: number) => v.toFixed(1)}
            width={55}
          />
          <Tooltip content={<CustomTooltip />} />
          <Legend
            wrapperStyle={{ fontSize: 11 }}
            formatter={(value: string) => (
              <span className="text-xs text-muted-foreground capitalize">{value}</span>
            )}
          />
          <Line
            type="monotone"
            dataKey="historical"
            stroke="#3b82f6"
            dot={false}
            strokeWidth={1.5}
            name="historical"
          />
          <Line
            type="monotone"
            dataKey="synthetic"
            stroke="#f59e0b"
            dot={false}
            strokeWidth={1.5}
            name="synthetic"
            strokeDasharray="4 2"
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}
