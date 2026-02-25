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
import { type DrawdownPoint } from "@/lib/types"

interface DrawdownChartProps {
  data: DrawdownPoint[]
  height?: number
}

export function DrawdownChart({ data, height = 400 }: DrawdownChartProps) {
  const displayData = useMemo(() => {
    const maxPoints = 300
    const step = Math.max(1, Math.floor(data.length / maxPoints))
    return data.filter((_, i) => i % step === 0).map((d) => ({
      ...d,
      historicalDdPct: d.historicalDrawdown * 100,
      syntheticDdPct: d.syntheticDrawdown * 100,
    }))
  }, [data])

  const { histMaxDd, synthMaxDd, histAvgDd, synthAvgDd } = useMemo(() => {
    const hDds = data.map((d) => d.historicalDrawdown)
    const sDds = data.map((d) => d.syntheticDrawdown)
    return {
      histMaxDd: Math.min(...hDds) * 100,
      synthMaxDd: Math.min(...sDds) * 100,
      histAvgDd: (hDds.reduce((a, b) => a + b, 0) / hDds.length) * 100,
      synthAvgDd: (sDds.reduce((a, b) => a + b, 0) / sDds.length) * 100,
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
              const isHist = entry.name === "historicalDdPct"
              return (
                <div key={entry.name} className="flex items-center justify-between gap-6 text-xs">
                  <span className="flex items-center gap-1.5">
                    <span
                      className="inline-block h-2 w-2 rounded-full"
                      style={{ backgroundColor: isHist ? "#ef4444" : "#f97316" }}
                    />
                    <span className="text-muted-foreground">
                      {isHist ? "Historical" : "Synthetic"}
                    </span>
                  </span>
                  <span className="font-mono font-medium text-red-500">
                    {entry.value.toFixed(2)}%
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
          <h4 className="text-sm font-semibold text-foreground">Drawdown Analysis</h4>
          <p className="text-xs text-muted-foreground">
            Peak-to-trough decline from running maximum
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-4 text-xs">
          <div className="flex items-center gap-1">
            <span className="inline-block h-2 w-2 rounded-full bg-red-500" />
            <span className="text-muted-foreground">Hist Max:</span>
            <span className="font-mono font-medium text-red-500">
              {histMaxDd.toFixed(2)}%
            </span>
          </div>
          <div className="flex items-center gap-1">
            <span className="inline-block h-2 w-2 rounded-full bg-orange-500" />
            <span className="text-muted-foreground">Synth Max:</span>
            <span className="font-mono font-medium text-red-500">
              {synthMaxDd.toFixed(2)}%
            </span>
          </div>
          <div className="flex items-center gap-1">
            <span className="text-muted-foreground">Hist Avg:</span>
            <span className="font-mono font-medium text-foreground">
              {histAvgDd.toFixed(2)}%
            </span>
          </div>
          <div className="flex items-center gap-1">
            <span className="text-muted-foreground">Synth Avg:</span>
            <span className="font-mono font-medium text-foreground">
              {synthAvgDd.toFixed(2)}%
            </span>
          </div>
        </div>
      </div>
      <ResponsiveContainer width="100%" height={height}>
        <AreaChart data={displayData} margin={{ top: 10, right: 10, bottom: 0, left: 10 }}>
          <defs>
            <linearGradient id="ddHistGrad" x1="0" y1="1" x2="0" y2="0">
              <stop offset="5%" stopColor="#ef4444" stopOpacity={0.25} />
              <stop offset="95%" stopColor="#ef4444" stopOpacity={0} />
            </linearGradient>
            <linearGradient id="ddSynthGrad" x1="0" y1="1" x2="0" y2="0">
              <stop offset="5%" stopColor="#f97316" stopOpacity={0.15} />
              <stop offset="95%" stopColor="#f97316" stopOpacity={0} />
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
                {value === "historicalDdPct" ? "Historical" : "Synthetic"}
              </span>
            )}
          />
          <ReferenceLine y={0} stroke="var(--muted-foreground)" strokeWidth={1} />
          <Area
            type="monotone"
            dataKey="historicalDdPct"
            stroke="#ef4444"
            fill="url(#ddHistGrad)"
            strokeWidth={1.5}
            dot={false}
            name="historicalDdPct"
          />
          <Area
            type="monotone"
            dataKey="syntheticDdPct"
            stroke="#f97316"
            fill="url(#ddSynthGrad)"
            strokeWidth={1.5}
            dot={false}
            name="syntheticDdPct"
            strokeDasharray="4 2"
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  )
}
