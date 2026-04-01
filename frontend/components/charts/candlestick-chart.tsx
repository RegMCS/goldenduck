"use client"

import { useMemo, useCallback, useState } from "react"
import {
  ComposedChart,
  Bar,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
  Cell,
  Brush,
} from "recharts"
import { MousePointer2 } from "lucide-react"
import { type OHLCVDataPoint } from "@/lib/types"
import { InfoTooltip } from "@/components/info-tooltip"

interface CandlestickChartProps {
  data: OHLCVDataPoint[]
  title: string
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
  isBullish: boolean
  idx: number
  ma: number
}

export function CandlestickChart({
  data,
  title,
  height = 400,
}: CandlestickChartProps) {
  const defaultPeriod = Math.max(2, Math.round(data.length * 0.05))
  const [showAvg, setShowAvg] = useState(true)
  const [showMA, setShowMA] = useState(true)
  const maPeriod = defaultPeriod

  const displayData = useMemo(() => {
    return data.map((d, idx, arr) => {
      const maStart = Math.max(0, idx - maPeriod + 1)
      const maSlice = arr.slice(maStart, idx + 1)
      const ma = maSlice.reduce((s, x) => s + x.close, 0) / maSlice.length

      return {
        date: d.date,
        open: d.open,
        high: d.high,
        low: d.low,
        close: d.close,
        volume: d.volume,
        bodyBottom: Math.min(d.open, d.close),
        bodyHeight: Math.abs(d.close - d.open),
        isBullish: d.close >= d.open,
        idx: idx + 1,
        ma: Math.round(ma * 100) / 100,
      }
    }) as CandleData[]
  }, [data, maPeriod])

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

  const CustomTooltip = useCallback(
    ({
      active,
      payload,
    }: {
      active?: boolean
      payload?: Array<{ payload: CandleData }>
    }) => {
      if (!active || !payload?.length) return null
      const d = payload[0].payload
      const changePercent = (((d.close - d.open) / d.open) * 100).toFixed(2)
      const isPositive = d.close >= d.open

      return (
        <div className="rounded-lg border border-border bg-card/95 backdrop-blur p-3 shadow-xl">
          <p className="mb-2 text-[10px] font-mono text-muted-foreground">
            Point #{d.idx}
          </p>
          <div className="grid grid-cols-2 gap-x-5 gap-y-1 text-xs">
            <span className="text-muted-foreground">Open</span>
            <span className="font-mono text-foreground text-right tabular-nums">
              {d.open.toFixed(2)}
            </span>
            <span className="text-muted-foreground">High</span>
            <span className="font-mono text-emerald-500 text-right tabular-nums">
              {d.high.toFixed(2)}
            </span>
            <span className="text-muted-foreground">Low</span>
            <span className="font-mono text-red-500 text-right tabular-nums">
              {d.low.toFixed(2)}
            </span>
            <span className="text-muted-foreground">Close</span>
            <span className="font-mono text-foreground text-right tabular-nums">
              {d.close.toFixed(2)}
            </span>
            <span className="text-muted-foreground">Volume</span>
            <span className="font-mono text-foreground text-right tabular-nums">
              {(d.volume / 1e6).toFixed(2)}M
            </span>
            <span className="text-muted-foreground">Change</span>
            <span
              className={`font-mono text-right font-semibold ${isPositive ? "text-emerald-500" : "text-red-500"}`}
            >
              {isPositive ? "+" : ""}
              {changePercent}%
            </span>
          </div>
        </div>
      )
    },
    []
  )

  const CandleShape = useCallback(
    (props: { x?: number; y?: number; width?: number; height?: number; payload?: CandleData }) => {
      const { x = 0, width = 0, payload } = props
      if (!payload) return null

      const { isBullish, open, close, high, low } = payload
      const fill = isBullish ? "#22c55e" : "#ef4444"
      const stroke = isBullish ? "#16a34a" : "#dc2626"

      const priceHeight = height * 0.6
      const range = maxPrice - minPrice
      const scale = (v: number) => priceHeight - ((v - minPrice) / range) * priceHeight

      const bodyTop = scale(Math.max(open, close))
      const bodyBottom = scale(Math.min(open, close))
      const wickTopY = scale(high)
      const wickBottomY = scale(low)
      const bodyH = Math.max(bodyBottom - bodyTop, 1)

      const centerX = x + width / 2
      const bodyWidth = Math.max(width * 0.7, 2)

      return (
        <g>
          <line
            x1={centerX}
            y1={wickTopY}
            x2={centerX}
            y2={wickBottomY}
            stroke={stroke}
            strokeWidth={1}
          />
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
  const totalChange = (((lastClose - firstClose) / firstClose) * 100).toFixed(2)
  const isPositiveTotal = lastClose >= firstClose

  const priceHeight = Math.round(height * 0.88)

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex items-start justify-between gap-4 pl-5 pr-5">
        <div>
          <h4 className="text-sm font-semibold text-foreground">{title}</h4>
          <p className="mt-0.5 flex items-center gap-1 text-[10px] text-muted-foreground">
            <MousePointer2 className="h-3 w-3 shrink-0" />
            Hover candles for full OHLCV detail
          </p>
        </div>
        <div className="flex items-center gap-3 text-xs shrink-0">
          <div className="flex items-center gap-1">
            <span className="text-muted-foreground">Last close:</span>
            <span className="font-mono font-semibold text-foreground">
              {lastClose.toFixed(2)}
            </span>
          </div>
          <div className="flex items-center gap-1">
            <span className="text-muted-foreground">Δ Close:</span>
            <span
              className={`font-mono font-semibold ${isPositiveTotal ? "text-emerald-500" : "text-red-500"}`}
            >
              {isPositiveTotal ? "+" : ""}
              {totalChange}%
            </span>
          </div>
        </div>
      </div>

      {/* Legend + toggles */}
      <div className="flex flex-col gap-3 pt-0 pl-5 pr-5">
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2">
            <span className="inline-block h-2.5 w-2.5 rounded-sm bg-emerald-500" />
            <span className="text-xs text-muted-foreground">Bull</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="inline-block h-2.5 w-2.5 rounded-sm bg-red-500" />
            <span className="text-xs text-muted-foreground">Bear</span>
          </div>
        </div>
        <div className="flex flex-wrap gap-3">
          <button
            onClick={() => setShowAvg((v) => !v)}
            className={`flex items-center gap-2 rounded border px-3 py-1 text-xs font-medium transition-colors ${
              showAvg
                ? "border-muted-foreground/40 bg-muted text-foreground"
                : "border-border bg-transparent text-muted-foreground opacity-50"
            }`}
          >
            <span
              className="inline-block w-5 border-t border-dashed"
              style={{ borderColor: showAvg ? "var(--muted-foreground)" : "var(--border)" }}
            />
            Mean Average
          </button>
          <div className="flex items-center gap-1.5">
            <button
              onClick={() => setShowMA((v) => !v)}
              className={`flex items-center gap-2 rounded border px-3 py-1 text-xs font-medium transition-colors ${
                showMA
                  ? "border-amber-500/50 bg-amber-50/60 text-amber-700 dark:bg-amber-950/20 dark:text-amber-400"
                  : "border-border bg-transparent text-muted-foreground opacity-50"
              }`}
            >
              <span
                className="inline-block w-5 border-t-2"
                style={{ borderColor: showMA ? "#f59e0b" : "var(--border)" }}
              />
              Moving Average
            </button>
            <InfoTooltip content={[
              "Moving Average",
              "Smooths price data by averaging the last ~5% of closing prices.",
              "Helps identify the overall trend direction by filtering out short-term noise.",
              "Note: Auto-scaled to ~5% of your data length.",
            ]} />
          </div>
        </div>
      </div>

      {/* Price chart */}
      <ResponsiveContainer width="100%" height={priceHeight} className="mt-2">
        <ComposedChart data={displayData} margin={{ top: 8, right: 68, bottom: 4, left: 5 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" opacity={0.4} />
          <XAxis
            dataKey="idx"
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
            width={52}
          />
          <Tooltip content={<CustomTooltip />} />
          <Brush
            dataKey="idx"
            height={28}
            stroke="var(--border)"
            fill="var(--muted)"
            travellerWidth={10}
            gap={1}
          />
          {showAvg && (
            <ReferenceLine
              y={avgPrice}
              stroke="var(--muted-foreground)"
              strokeDasharray="5 5"
              opacity={0.6}
              label={(props: { viewBox?: { x?: number; y?: number; width?: number } }) => {
                const { x = 0, y = 0, width = 0 } = props.viewBox ?? {}
                const right = x + width
                return (
                  <g>
                    <text
                      x={x - 6}
                      y={y}
                      textAnchor="end"
                      dominantBaseline="middle"
                      fill="var(--muted-foreground)"
                      fontSize={9}
                      fontWeight={600}
                    >
                      {avgPrice.toFixed(1)}
                    </text>
                    <text
                      x={right + 16}
                      y={y}
                      textAnchor="start"
                      dominantBaseline="middle"
                      fill="var(--muted-foreground)"
                      fontSize={9}
                      fontWeight={500}
                    >
                      Mean Avg
                    </text>
                  </g>
                )
              }}
            />
          )}
          {showMA && (
            <Line
              type="monotone"
              dataKey="ma"
              stroke="#f59e0b"
              strokeWidth={1.5}
              dot={false}
              isAnimationActive={false}
              name={`MA ${maPeriod}`}
              label={(props: { x?: number; y?: number; index?: number }) => {
                if (props.index !== displayData.length - 1) return <g />
                return (
                  <text
                    x={(props.x ?? 0) + 63}
                    y={props.y ?? 0}
                    textAnchor="end"
                    fill="#f59e0b"
                    fontSize={10}
                    fontWeight={600}
                    dominantBaseline="middle"
                  >
                    MA {maPeriod}
                  </text>
                )
              }}
            />
          )}
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
      <p className="text-center text-[10px] text-muted-foreground/60 -mt-1">
        Drag handles to zoom · scroll to pan
      </p>
    </div>
  )
}
