"use client"

import { useMemo, useCallback } from "react"
import {
  ComposedChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
  Cell,
} from "recharts"
import { type OHLCVDataPoint } from "@/lib/types"

interface CandlestickChartProps {
  data: OHLCVDataPoint[]
  title: string
  subtitle?: string
  height?: number
}

interface CandleData {
  date: string
  open: number
  high: number
  low: number
  close: number
  volume: number
  bodyBottom: number
  bodyHeight: number
  wickTop: number
  wickBottom: number
  isBullish: boolean
  idx: number
}

export function CandlestickChart({
  data,
  title,
  subtitle,
  height = 400,
}: CandlestickChartProps) {
  const displayData = useMemo(() => {
    // Downsample to reasonable number for display
    const maxPoints = 120
    const step = Math.max(1, Math.floor(data.length / maxPoints))
    const sampled = data.filter((_, i) => i % step === 0)

    return sampled.map((d, idx) => {
      const isBullish = d.close >= d.open
      return {
        date: d.date,
        open: d.open,
        high: d.high,
        low: d.low,
        close: d.close,
        volume: d.volume,
        bodyBottom: Math.min(d.open, d.close),
        bodyHeight: Math.abs(d.close - d.open),
        wickTop: d.high,
        wickBottom: d.low,
        isBullish,
        idx,
      } as CandleData
    })
  }, [data])

  const { minPrice, maxPrice, avgPrice } = useMemo(() => {
    const lows = displayData.map((d) => d.low)
    const highs = displayData.map((d) => d.high)
    const min = Math.min(...lows)
    const max = Math.max(...highs)
    const avg = displayData.reduce((s, d) => s + d.close, 0) / displayData.length
    return {
      minPrice: min - (max - min) * 0.05,
      maxPrice: max + (max - min) * 0.05,
      avgPrice: Math.round(avg * 100) / 100,
    }
  }, [displayData])

  const formatDate = useCallback((value: string) => {
    if (!value) return ""
    const parts = value.split("-")
    return `${parts[1]}/${parts[2]}`
  }, [])

  const CustomTooltip = useCallback(
    ({ active, payload }: { active?: boolean; payload?: Array<{ payload: CandleData }> }) => {
      if (!active || !payload?.length) return null
      const d = payload[0].payload
      const changePercent = ((d.close - d.open) / d.open * 100).toFixed(2)
      const isPositive = d.close >= d.open

      return (
        <div className="rounded-lg border border-border bg-card p-3 shadow-lg">
          <p className="mb-2 text-xs font-mono text-muted-foreground">{d.date}</p>
          <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-xs">
            <span className="text-muted-foreground">Open</span>
            <span className="font-mono text-foreground text-right">{d.open.toFixed(2)}</span>
            <span className="text-muted-foreground">High</span>
            <span className="font-mono text-foreground text-right">{d.high.toFixed(2)}</span>
            <span className="text-muted-foreground">Low</span>
            <span className="font-mono text-foreground text-right">{d.low.toFixed(2)}</span>
            <span className="text-muted-foreground">Close</span>
            <span className="font-mono text-foreground text-right">{d.close.toFixed(2)}</span>
            <span className="text-muted-foreground">Volume</span>
            <span className="font-mono text-foreground text-right">{(d.volume / 1e6).toFixed(2)}M</span>
            <span className="text-muted-foreground">Change</span>
            <span
              className={`font-mono text-right font-medium ${isPositive ? "text-emerald-500" : "text-red-500"}`}
            >
              {isPositive ? "+" : ""}{changePercent}%
            </span>
          </div>
        </div>
      )
    },
    []
  )

  // Custom candle shape
  const CandleShape = useCallback(
    (props: {
      x?: number
      y?: number
      width?: number
      height?: number
      payload?: CandleData
    }) => {
      const { x = 0, width = 0, payload } = props
      if (!payload) return null

      const { isBullish, open, close, high, low } = payload
      const fill = isBullish ? "#22c55e" : "#ef4444"
      const stroke = isBullish ? "#16a34a" : "#dc2626"

      // Calculate pixel positions from domain
      const chartHeight = height - 60 // approximate internal chart area
      const range = maxPrice - minPrice
      const scale = (v: number) => chartHeight - ((v - minPrice) / range) * chartHeight + 30

      const bodyTop = scale(Math.max(open, close))
      const bodyBottom = scale(Math.min(open, close))
      const wickTopY = scale(high)
      const wickBottomY = scale(low)
      const bodyH = Math.max(bodyBottom - bodyTop, 1)

      const centerX = x + width / 2
      const bodyWidth = Math.max(width * 0.7, 2)

      return (
        <g>
          {/* Wick */}
          <line
            x1={centerX}
            y1={wickTopY}
            x2={centerX}
            y2={wickBottomY}
            stroke={stroke}
            strokeWidth={1}
          />
          {/* Body */}
          <rect
            x={centerX - bodyWidth / 2}
            y={bodyTop}
            width={bodyWidth}
            height={bodyH}
            fill={fill}
            stroke={stroke}
            strokeWidth={0.5}
            rx={0.5}
          />
        </g>
      )
    },
    [height, minPrice, maxPrice]
  )

  const firstClose = displayData[0]?.close ?? 0
  const lastClose = displayData[displayData.length - 1]?.close ?? 0
  const totalChange = ((lastClose - firstClose) / firstClose * 100).toFixed(2)
  const isPositiveTotal = lastClose >= firstClose

  return (
    <div className="space-y-3">
      <div className="flex items-baseline justify-between">
        <div>
          <h4 className="text-sm font-semibold text-foreground">{title}</h4>
          {subtitle && (
            <p className="text-xs text-muted-foreground">{subtitle}</p>
          )}
        </div>
        <div className="flex items-center gap-3 text-xs">
          <div className="flex items-center gap-1">
            <span className="text-muted-foreground">Last:</span>
            <span className="font-mono font-medium text-foreground">{lastClose.toFixed(2)}</span>
          </div>
          <div className="flex items-center gap-1">
            <span className="text-muted-foreground">Chg:</span>
            <span
              className={`font-mono font-medium ${isPositiveTotal ? "text-emerald-500" : "text-red-500"}`}
            >
              {isPositiveTotal ? "+" : ""}{totalChange}%
            </span>
          </div>
          <div className="flex items-center gap-2">
            <span className="flex items-center gap-1">
              <span className="inline-block h-2 w-2 rounded-sm bg-emerald-500" />
              <span className="text-muted-foreground">Bull</span>
            </span>
            <span className="flex items-center gap-1">
              <span className="inline-block h-2 w-2 rounded-sm bg-red-500" />
              <span className="text-muted-foreground">Bear</span>
            </span>
          </div>
        </div>
      </div>
      <ResponsiveContainer width="100%" height={height}>
        <ComposedChart data={displayData} margin={{ top: 10, right: 10, bottom: 0, left: 10 }}>
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
            domain={[minPrice, maxPrice]}
            tick={{ fontSize: 10, fill: "var(--muted-foreground)" }}
            tickLine={false}
            axisLine={{ stroke: "var(--border)" }}
            tickFormatter={(v: number) => v.toFixed(1)}
            width={55}
          />
          <Tooltip content={<CustomTooltip />} />
          <ReferenceLine
            y={avgPrice}
            stroke="var(--muted-foreground)"
            strokeDasharray="5 5"
            opacity={0.4}
            label={{
              value: `Avg: ${avgPrice.toFixed(2)}`,
              position: "right",
              fill: "var(--muted-foreground)",
              fontSize: 9,
            }}
          />
          {/* Invisible bar used to trigger tooltip and provide x positioning */}
          <Bar
            dataKey="bodyHeight"
            stackId="candle"
            shape={<CandleShape />}
            isAnimationActive={false}
          >
            {displayData.map((entry) => (
              <Cell key={`cell-${entry.idx}`} fill="transparent" />
            ))}
          </Bar>
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  )
}
