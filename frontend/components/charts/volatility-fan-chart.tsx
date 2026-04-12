"use client"

import { useMemo, useCallback } from "react"
import { Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Area, ComposedChart, Brush } from "recharts"

interface VolatilityFanPoint {
  date: string
  timestamp: number
  historical: number
  p10: number
  p50: number
  p90: number
}

interface VolatilityFanChartProps {
  data: VolatilityFanPoint[]
  height?: number
}

const SERIES = [
  { key: "p50", label: "Median (50th)", color: "#f59e0b" },
  { key: "p10", label: "10th percentile", color: "#f59e0b" },
  { key: "p90", label: "90th percentile", color: "#f59e0b" },
  { key: "historical", label: "Historical", color: "#3b82f6" },
]

export function VolatilityFanChart({ data, height = 420 }: VolatilityFanChartProps) {
  const processedData = useMemo(() => {
    if (!data || data.length === 0) return []

    return data.map((point, i) => ({
      idx: i + 1,
      timestamp: point.timestamp,
      date: point.date,
      historical: parseFloat(point.historical.toFixed(2)),
      p10: parseFloat(point.p10.toFixed(2)),
      p50: parseFloat(point.p50.toFixed(2)),
      p90: parseFloat(point.p90.toFixed(2)),
    }))
  }, [data])

  const bandWidth = useMemo(() => {
    if (processedData.length === 0) return 0
    const lastPoint = processedData[processedData.length - 1]
    return (lastPoint.p90 - lastPoint.p10).toFixed(2)
  }, [processedData])

  const CustomTooltip = useCallback(({
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
          {SERIES.map(({ key, label: seriesLabel, color }) => {
            const entry = payload.find((p) => p.name === seriesLabel)
            if (!entry) return null
            return (
              <div key={key} className="flex items-center justify-between gap-8 text-xs">
                <span className="flex items-center gap-1.5">
                  <span className="inline-block h-2 w-2 rounded-full" style={{ backgroundColor: color }} />
                  <span className="text-muted-foreground">{seriesLabel}</span>
                </span>
                <span className="font-mono font-semibold text-foreground">
                  {entry.value.toFixed(2)}
                </span>
              </div>
            )
          })}
        </div>
      </div>
    )
  }, [])

  return (
    <div className="w-full space-y-4">
      <div className="flex items-start justify-between">
        <div>
          <h4 className="text-sm font-semibold text-foreground">Volatility Fan Chart</h4>
        </div>
        <div className="text-xs text-muted-foreground shrink-0">
          Band width at horizon: <span className="font-mono font-semibold">{bandWidth}</span>
        </div>
      </div>
      {/* Custom legend */}
      <div className="flex flex-wrap items-center gap-4 text-xs">
        <div className="flex items-center gap-1.5">
          <span className="inline-block w-5 border-t-2 border-amber-500" />
          <span className="text-muted-foreground">Median (50th)</span>
        </div>
        <div className="flex items-center gap-1.5">
          <span className="inline-block w-5 border-t border-dashed border-amber-500" />
          <span className="text-muted-foreground">10th percentile</span>
        </div>
        <div className="flex items-center gap-1.5">
          <span className="inline-block w-5 border-t border-dashed border-amber-500" />
          <span className="text-muted-foreground">90th percentile</span>
        </div>
        <div className="flex items-center gap-1.5">
          <span className="inline-block h-2.5 w-2.5 rounded-full bg-blue-500" />
          <span className="text-muted-foreground">Historical</span>
        </div>
      </div>
      <ResponsiveContainer width="100%" height={height}>
        <ComposedChart
          data={processedData}
          margin={{ top: 5, right: 30, left: 10, bottom: 29 }}
        >
          <defs>
            <linearGradient id="bandGradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#f59e0b" stopOpacity={0.15} />
              <stop offset="100%" stopColor="#f59e0b" stopOpacity={0.05} />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
          <XAxis
            dataKey="idx"
            tick={{ fontSize: 10, fill: "var(--muted-foreground)" }}
            tickLine={false}
            axisLine={{ stroke: "var(--border)" }}
            interval={Math.max(1, Math.floor(processedData.length / 8))}
            label={{ value: "Data points", position: "insideBottomRight", offset: 0, dy: 18, fontSize: 9, fill: "var(--muted-foreground)" }}
          />
          <YAxis
            tick={{ fontSize: 10, fill: "var(--muted-foreground)" }}
            tickLine={false}
            axisLine={{ stroke: "var(--border)" }}
            width={52}
            label={{ value: "Price (indexed to 100)", angle: -90, position: "insideLeft", offset: 10, fontSize: 10, fill: "var(--muted-foreground)" }}
          />
          <Tooltip content={<CustomTooltip />} />

          {/* 10th-90th percentile band */}
          <Area
            type="monotone"
            dataKey="p90"
            stroke="transparent"
            fill="url(#bandGradient)"
            isAnimationActive={false}
            name="10th-90th band"
          />

          {/* 50th percentile (median) */}
          <Line
            type="monotone"
            dataKey="p50"
            stroke="#f59e0b"
            strokeWidth={2}
            dot={false}
            isAnimationActive={false}
            name="Median (50th)"
          />

          {/* 10th percentile */}
          <Line
            type="monotone"
            dataKey="p10"
            stroke="#f59e0b"
            strokeWidth={1}
            strokeDasharray="5 5"
            dot={false}
            isAnimationActive={false}
            name="10th percentile"
          />

          {/* 90th percentile */}
          <Line
            type="monotone"
            dataKey="p90"
            stroke="#f59e0b"
            strokeWidth={1}
            strokeDasharray="5 5"
            dot={false}
            isAnimationActive={false}
            name="90th percentile"
          />

          {/* Historical price overlay */}
          <Line
            type="monotone"
            dataKey="historical"
            stroke="#3b82f6"
            strokeWidth={2}
            dot={false}
            isAnimationActive={false}
            name="Historical"
          />
          <Brush
            dataKey="idx"
            height={28}
            stroke="var(--border)"
            fill="var(--card)"
            travellerWidth={8}
          />
        </ComposedChart>
      </ResponsiveContainer>
      <p className="-mt-6 text-center text-[10px] text-muted-foreground/60">Drag handles to zoom · scroll to pan</p>
      <div className="text-xs text-muted-foreground space-y-1">
        <p>
          <span className="font-semibold">Fan interpretation:</span> Wider band indicates higher volatility. Compare historical price (blue) to synthetic paths (orange band).
        </p>
      </div>
    </div>
  )
}
