"use client"

import { useMemo } from "react"
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, Area, AreaChart, ComposedChart } from "recharts"

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

export function VolatilityFanChart({ data, height = 420 }: VolatilityFanChartProps) {
  const processedData = useMemo(() => {
    if (!data || data.length === 0) return []

    return data.map((point) => ({
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

  return (
    <div className="w-full space-y-4">
      <div className="flex items-center justify-between">
        <h4 className="text-sm font-semibold text-foreground">Volatility Fan Chart</h4>
        <div className="text-xs text-muted-foreground">
          Band width at horizon: <span className="font-mono font-semibold">{bandWidth}</span>
        </div>
      </div>
      <ResponsiveContainer width="100%" height={height}>
        <ComposedChart
          data={processedData}
          margin={{ top: 5, right: 30, left: 0, bottom: 5 }}
        >
          <defs>
            <linearGradient id="bandGradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#f59e0b" stopOpacity={0.15} />
              <stop offset="100%" stopColor="#f59e0b" stopOpacity={0.05} />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
          <XAxis
            dataKey="date"
            tick={{ fontSize: 12 }}
            stroke="var(--muted-foreground)"
            interval={Math.max(0, Math.floor(processedData.length / 10))}
          />
          <YAxis
            tick={{ fontSize: 12 }}
            stroke="var(--muted-foreground)"
            label={{ value: "Price (indexed to 100)", angle: -90, position: "insideLeft" }}
          />
          <Tooltip
            contentStyle={{
              backgroundColor: "var(--background)",
              border: "1px solid var(--border)",
              borderRadius: "0.5rem",
            }}
            formatter={(value: number) => value.toFixed(2)}
            labelStyle={{ color: "var(--foreground)" }}
          />
          <Legend
            wrapperStyle={{ fontSize: 12 }}
            verticalAlign="bottom"
            height={36}
          />

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
        </ComposedChart>
      </ResponsiveContainer>
      <div className="text-xs text-muted-foreground space-y-1">
        <p>
          <span className="font-semibold">Fan interpretation:</span> Wider band indicates higher volatility. Compare historical price (blue) to synthetic paths (orange band).
        </p>
      </div>
    </div>
  )
}
