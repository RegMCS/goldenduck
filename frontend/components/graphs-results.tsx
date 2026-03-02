"use client"

import { useMemo } from "react"
import { type GeneratedData } from "@/lib/types"
import { CandlestickChart } from "@/components/charts/candlestick-chart"
import { PriceOverlayChart } from "@/components/charts/price-overlay-chart"
import { CumulativeReturnChart } from "@/components/charts/cumulative-return-chart"
import { DrawdownChart } from "@/components/charts/drawdown-chart"
import { StatsPanel } from "@/components/charts/stats-panel"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"

interface VisualizationResultsProps {
  data: GeneratedData
}

function FidelityGauge({ score }: { score: number }) {
  const r = 28
  const circ = 2 * Math.PI * r
  const offset = circ * (1 - score / 100)
  const color = score >= 75 ? "#22c55e" : score >= 50 ? "#f59e0b" : "#ef4444"
  const label = score >= 75 ? "Excellent match" : score >= 50 ? "Moderate match" : "Low match"

  return (
    <div className="flex items-center gap-3 py-1">
      <div className="relative h-16 w-16 shrink-0">
        <svg viewBox="0 0 64 64" className="h-16 w-16 -rotate-90">
          <circle cx="32" cy="32" r={r} fill="none" stroke="var(--border)" strokeWidth="6" />
          <circle
            cx="32"
            cy="32"
            r={r}
            fill="none"
            stroke={color}
            strokeWidth="6"
            strokeDasharray={circ}
            strokeDashoffset={offset}
            strokeLinecap="round"
            style={{ transition: "stroke-dashoffset 1.2s ease-out" }}
          />
        </svg>
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <span className="text-sm font-bold tabular-nums leading-none" style={{ color }}>
            {score}
          </span>
          <span className="text-[8px] text-muted-foreground uppercase tracking-wider mt-0.5">
            score
          </span>
        </div>
      </div>
      <div>
        <p className="text-sm font-semibold text-foreground">Fidelity</p>
        <p className="text-[11px] font-medium" style={{ color }}>
          {label}
        </p>
        <p className="text-[10px] text-muted-foreground">vs. historical</p>
      </div>
    </div>
  )
}

function DualMetricCard({
  label,
  histValue,
  synthValue,
  histClass,
  synthClass,
  delta,
}: {
  label: string
  histValue: string
  synthValue: string
  histClass: string
  synthClass: string
  delta: string
}) {
  return (
    <div className="rounded-xl border border-border bg-card p-4 space-y-3 min-w-0">
      <p className="text-[10px] font-semibold text-muted-foreground uppercase tracking-widest truncate">
        {label}
      </p>
      <div className="flex items-stretch gap-2">
        <div className="flex-1 min-w-0 space-y-0.5">
          <p className="text-[10px] text-blue-500 font-medium">Hist</p>
          <p className={`text-sm font-mono font-bold tabular-nums truncate ${histClass}`}>
            {histValue}
          </p>
        </div>
        <div className="w-px bg-border self-stretch" />
        <div className="flex-1 min-w-0 space-y-0.5 text-right">
          <p className="text-[10px] text-amber-500 font-medium">Synth</p>
          <p className={`text-sm font-mono font-bold tabular-nums truncate ${synthClass}`}>
            {synthValue}
          </p>
        </div>
      </div>
      <p className="text-[10px] text-muted-foreground truncate">Δ {delta}</p>
    </div>
  )
}

export function VisualizationResults({ data }: VisualizationResultsProps) {
  const h = data.stats.historical
  const s = data.stats.synthetic

  const fidelityScore = useMemo(() => {
    const safe = (a: number, b: number) =>
      1 - Math.min(1, Math.abs((a - b) / Math.max(Math.abs(b), 0.01)))
    return Math.round(
      ((safe(s.annualizedVol, h.annualizedVol) +
        safe(s.kurtosis, h.kurtosis) +
        safe(s.skewness, h.skewness) +
        safe(s.sharpe, h.sharpe)) /
        4) *
        100
    )
  }, [h, s])

  const pct = (v: number) => `${(v * 100).toFixed(2)}%`
  const num = (v: number) => v.toFixed(3)
  const rc = (v: number) => (v >= 0 ? "text-emerald-500" : "text-red-500")

  return (
    <div className="space-y-8">
      {/* Fidelity gauge + key metric cards */}
      <div className="grid gap-3 grid-cols-2 md:grid-cols-3 xl:grid-cols-5">
        <div className="rounded-xl border border-border bg-card px-5 py-4 col-span-2 md:col-span-1 flex items-center">
          <FidelityGauge score={fidelityScore} />
        </div>
        <DualMetricCard
          label="Ann. Return"
          histValue={pct(h.annualizedReturn)}
          synthValue={pct(s.annualizedReturn)}
          histClass={rc(h.annualizedReturn)}
          synthClass={rc(s.annualizedReturn)}
          delta={`${((s.annualizedReturn - h.annualizedReturn) * 100).toFixed(2)}pp`}
        />
        <DualMetricCard
          label="Ann. Volatility"
          histValue={pct(h.annualizedVol)}
          synthValue={pct(s.annualizedVol)}
          histClass="text-foreground"
          synthClass="text-foreground"
          delta={`${h.annualizedVol > 0 ? (s.annualizedVol / h.annualizedVol).toFixed(2) : "—"}× ratio`}
        />
        <DualMetricCard
          label="Sharpe Ratio"
          histValue={num(h.sharpe)}
          synthValue={num(s.sharpe)}
          histClass={rc(h.sharpe)}
          synthClass={rc(s.sharpe)}
          delta={(s.sharpe - h.sharpe).toFixed(3)}
        />
        <DualMetricCard
          label="Max Drawdown"
          histValue={pct(h.maxDrawdown)}
          synthValue={pct(s.maxDrawdown)}
          histClass="text-red-500"
          synthClass="text-red-500"
          delta={`${((s.maxDrawdown - h.maxDrawdown) * 100).toFixed(2)}pp`}
        />
      </div>

      {/* Candlestick comparison */}
      <section className="rounded-xl border border-border bg-card p-5">
        <div className="mb-5">
          <h3 className="text-base font-semibold text-foreground">OHLCV Candlestick Comparison</h3>
          <p className="mt-0.5 text-xs text-muted-foreground">
            Historical (left) vs synthetic (right) · Hover candles for full OHLCV detail · Volume bars below each chart
          </p>
        </div>
        <div className="grid gap-4 lg:grid-cols-2">
          <div className="rounded-lg border border-blue-500/20 bg-background p-3">
            <CandlestickChart
              data={data.historical}
              title="Historical"
              subtitle={`${data.historical.length} pts · ${data.historical[0]?.date} → ${data.historical[data.historical.length - 1]?.date}`}
              height={320}
            />
          </div>
          <div className="rounded-lg border border-amber-500/20 bg-background p-3">
            <CandlestickChart
              data={data.synthetic}
              title="Synthetic"
              subtitle={`${data.synthetic.length} pts · Generated`}
              height={320}
            />
          </div>
        </div>
      </section>

      {/* Analytics tabs — brush-zoomable */}
      <section className="space-y-4">
        <h3 className="text-base font-semibold text-foreground">Analytics</h3>
        <Tabs defaultValue="overlay">
          <TabsList className="grid w-full max-w-md grid-cols-3">
            <TabsTrigger value="overlay">Price Overlay</TabsTrigger>
            <TabsTrigger value="returns">Cum. Returns</TabsTrigger>
            <TabsTrigger value="drawdown">Drawdowns</TabsTrigger>
          </TabsList>
          <TabsContent value="overlay" className="mt-4">
            <div className="rounded-xl border border-border bg-card p-5">
              <PriceOverlayChart data={data.timeSeries} height={420} />
            </div>
          </TabsContent>
          <TabsContent value="returns" className="mt-4">
            <div className="rounded-xl border border-border bg-card p-5">
              <CumulativeReturnChart data={data.returns} height={420} />
            </div>
          </TabsContent>
          <TabsContent value="drawdown" className="mt-4">
            <div className="rounded-xl border border-border bg-card p-5">
              <DrawdownChart data={data.drawdowns} height={420} />
            </div>
          </TabsContent>
        </Tabs>
      </section>

      {/* Statistical comparison */}
      <section>
        <h3 className="mb-4 text-base font-semibold text-foreground">Statistical Comparison</h3>
        <StatsPanel stats={data.stats} />
      </section>
    </div>
  )
}
