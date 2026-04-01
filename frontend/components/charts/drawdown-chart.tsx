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
  ReferenceLine,
  Brush,
} from "recharts"
import { type DrawdownPoint } from "@/lib/types"

interface DrawdownChartProps {
  data: DrawdownPoint[]
  height?: number
}

export function DrawdownChart({ data, height = 400 }: DrawdownChartProps) {
  const displayData = useMemo(() => data.map((d, i) => ({
    ...d,
    idx: i + 1,
    historicalDdPct: d.historicalDrawdown * 100,
    syntheticDdPct: d.syntheticDrawdown * 100,
  })), [data])

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
            {payload.map((entry) => {
              const isHist = entry.name === "historicalDdPct"
              return (
                <div
                  key={entry.name}
                  className="flex items-center justify-between gap-8 text-xs"
                >
                  <span className="flex items-center gap-1.5">
                    <span
                      className="inline-block h-2 w-2 rounded-full"
                      style={{ backgroundColor: isHist ? "#ef4444" : "#f97316" }}
                    />
                    <span className="text-muted-foreground">
                      {isHist ? "Historical" : "Synthetic"}
                    </span>
                  </span>
                  <span className="font-mono font-semibold text-red-500">
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
    <div className="space-y-4">
      <div className="flex items-start justify-between gap-4">
        <div>
          <h4 className="text-sm font-semibold text-foreground">Drawdown Analysis</h4>
          <p className="text-xs text-muted-foreground mt-0.5">
            Peak-to-trough decline · dotted lines mark max drawdown
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-3 text-xs shrink-0">
          <div className="flex items-center gap-1.5">
            <span className="inline-block h-2.5 w-2.5 rounded-full bg-red-500" />
            <span className="text-muted-foreground">Hist max:</span>
            <span className="font-mono font-semibold text-red-500">
              {histMaxDd.toFixed(2)}%
            </span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="inline-block h-2.5 w-2.5 rounded-full bg-orange-500" />
            <span className="text-muted-foreground">Synth max:</span>
            <span className="font-mono font-semibold text-red-500">
              {synthMaxDd.toFixed(2)}%
            </span>
          </div>
          <div className="flex items-center gap-1 border-l border-border pl-3">
            <span className="text-muted-foreground">H avg:</span>
            <span className="font-mono font-semibold text-foreground">
              {histAvgDd.toFixed(2)}%
            </span>
          </div>
          <div className="flex items-center gap-1">
            <span className="text-muted-foreground">S avg:</span>
            <span className="font-mono font-semibold text-foreground">
              {synthAvgDd.toFixed(2)}%
            </span>
          </div>
        </div>
      </div>

      <ResponsiveContainer width="100%" height={height}>
        <AreaChart data={displayData} margin={{ top: 8, right: 12, bottom: 24, left: 10 }}>
          <defs>
            <linearGradient id="ddHistGrad" x1="0" y1="1" x2="0" y2="0">
              <stop offset="5%" stopColor="#ef4444" stopOpacity={0.35} />
              <stop offset="95%" stopColor="#ef4444" stopOpacity={0} />
            </linearGradient>
            <linearGradient id="ddSynthGrad" x1="0" y1="1" x2="0" y2="0">
              <stop offset="5%" stopColor="#f97316" stopOpacity={0.25} />
              <stop offset="95%" stopColor="#f97316" stopOpacity={0} />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" opacity={0.4} />
          <XAxis
            dataKey="idx"
            tick={{ fontSize: 10, fill: "var(--muted-foreground)" }}
            tickLine={false}
            axisLine={{ stroke: "var(--border)" }}
            interval={Math.max(1, Math.floor(displayData.length / 8))}
            label={{ value: "Data points", position: "insideBottomRight", offset: 0, dy: 18, fontSize: 9, fill: "var(--muted-foreground)" }}
          />
          <YAxis
            tick={{ fontSize: 10, fill: "var(--muted-foreground)" }}
            tickLine={false}
            axisLine={{ stroke: "var(--border)" }}
            tickFormatter={(v: number) => `${v.toFixed(0)}%`}
            width={52}
          />
          <Tooltip content={<CustomTooltip />} />
          <ReferenceLine y={0} stroke="var(--muted-foreground)" strokeWidth={1} opacity={0.5} />
          {/* Max drawdown annotation lines */}
          <ReferenceLine
            y={histMaxDd}
            stroke="#ef4444"
            strokeDasharray="5 3"
            opacity={0.7}
            label={{
              value: `H: ${histMaxDd.toFixed(1)}%`,
              position: "insideBottomRight",
              fill: "#ef4444",
              fontSize: 9,
            }}
          />
          <ReferenceLine
            y={synthMaxDd}
            stroke="#f97316"
            strokeDasharray="5 3"
            opacity={0.7}
            label={{
              value: `S: ${synthMaxDd.toFixed(1)}%`,
              position: "insideTopRight",
              fill: "#f97316",
              fontSize: 9,
            }}
          />
          <Brush
            dataKey="idx"
            height={28}
            stroke="var(--border)"
            fill="var(--card)"
            travellerWidth={8}
          />
          <Area
            type="monotone"
            dataKey="historicalDdPct"
            stroke="#ef4444"
            strokeWidth={2}
            fill="url(#ddHistGrad)"
            dot={false}
            name="historicalDdPct"
          />
          <Area
            type="monotone"
            dataKey="syntheticDdPct"
            stroke="#f97316"
            strokeWidth={2}
            fill="url(#ddSynthGrad)"
            dot={false}
            name="syntheticDdPct"
            strokeDasharray="5 2"
          />
        </AreaChart>
      </ResponsiveContainer>
      <p className="-mt-3 text-center text-[10px] text-muted-foreground/60">Drag handles to zoom · scroll to pan</p>
    </div>
  )
}
