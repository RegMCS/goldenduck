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
  Brush,
  ReferenceLine,
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
    let cov = 0,
      hVar = 0,
      sVar = 0
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
    ({
      active,
      payload,
      label,
    }: {
      active?: boolean
      payload?: Array<{ value: number; name: string }>
      label?: string
    }) => {
      if (!active || !payload?.length) return null
      return (
        <div className="rounded-lg border border-border bg-card/95 backdrop-blur p-3 shadow-xl">
          <p className="mb-2 text-xs font-mono text-muted-foreground">{label}</p>
          <div className="space-y-1.5">
            {payload.map((entry) => (
              <div key={entry.name} className="flex items-center justify-between gap-8 text-xs">
                <span className="flex items-center gap-1.5">
                  <span
                    className="inline-block h-2 w-2 rounded-full"
                    style={{
                      backgroundColor: entry.name === "historical" ? "#3b82f6" : "#f59e0b",
                    }}
                  />
                  <span className="text-muted-foreground capitalize">{entry.name}</span>
                </span>
                <span className="font-mono font-semibold text-foreground">
                  {entry.value.toFixed(2)}
                </span>
              </div>
            ))}
            {payload.length === 2 && (
              <div className="flex items-center justify-between gap-8 text-xs border-t border-border pt-1.5 mt-1">
                <span className="text-muted-foreground">Spread</span>
                <span
                  className={`font-mono font-semibold ${payload[0].value >= payload[1].value ? "text-blue-500" : "text-amber-500"}`}
                >
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

  const corrColor =
    Math.abs(correlation) >= 0.8
      ? "text-emerald-500"
      : Math.abs(correlation) >= 0.5
        ? "text-amber-500"
        : "text-red-500"

  return (
    <div className="space-y-4">
      <div className="flex items-start justify-between gap-4">
        <div>
          <h4 className="text-sm font-semibold text-foreground">Price Time Series Overlay</h4>
          <p className="text-xs text-muted-foreground mt-0.5">
            Historical vs synthetic · normalised to 100 · drag the brush below to zoom
          </p>
        </div>
        <div className="flex items-center gap-4 text-xs shrink-0">
          <div className="flex items-center gap-1.5">
            <span className="inline-block h-2.5 w-2.5 rounded-full bg-blue-500" />
            <span className="text-muted-foreground">Historical</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="inline-block h-2.5 w-2.5 rounded-full bg-amber-500" />
            <span className="text-muted-foreground">Synthetic</span>
          </div>
          <div className="flex items-center gap-1 border-l border-border pl-4">
            <span className="text-muted-foreground">Pearson r:</span>
            <span className={`font-mono font-semibold ${corrColor}`}>
              {correlation.toFixed(3)}
            </span>
          </div>
        </div>
      </div>

      <ResponsiveContainer width="100%" height={height}>
        <AreaChart data={displayData} margin={{ top: 8, right: 12, bottom: 0, left: 10 }}>
          <defs>
            <linearGradient id="histGradOverlay" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.25} />
              <stop offset="95%" stopColor="#3b82f6" stopOpacity={0.02} />
            </linearGradient>
            <linearGradient id="synthGradOverlay" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#f59e0b" stopOpacity={0.2} />
              <stop offset="95%" stopColor="#f59e0b" stopOpacity={0.02} />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" opacity={0.4} />
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
            tickFormatter={(v: number) => v.toFixed(0)}
            width={48}
          />
          <Tooltip content={<CustomTooltip />} />
          <ReferenceLine
            y={100}
            stroke="var(--muted-foreground)"
            strokeDasharray="4 4"
            opacity={0.35}
          />
          <Brush
            dataKey="date"
            height={28}
            stroke="var(--border)"
            fill="var(--card)"
            travellerWidth={8}
            tickFormatter={formatDate}
          />
          <Area
            type="monotone"
            dataKey="historical"
            stroke="#3b82f6"
            strokeWidth={2}
            fill="url(#histGradOverlay)"
            dot={false}
            name="historical"
          />
          <Area
            type="monotone"
            dataKey="synthetic"
            stroke="#f59e0b"
            strokeWidth={2}
            fill="url(#synthGradOverlay)"
            dot={false}
            name="synthetic"
            strokeDasharray="5 2"
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  )
}
