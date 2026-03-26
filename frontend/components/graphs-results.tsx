"use client"

import { useMemo } from "react"
import { type GeneratedData } from "@/lib/types"
import { CandlestickChart } from "@/components/charts/candlestick-chart"
import { PriceOverlayChart } from "@/components/charts/price-overlay-chart"
import { CumulativeReturnChart } from "@/components/charts/cumulative-return-chart"
import { DrawdownChart } from "@/components/charts/drawdown-chart"
import { VolatilityFanChart } from "@/components/charts/volatility-fan-chart"
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
        <p className="text-[10px] text-muted-foreground">vs. desired</p>
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
  // Guard against missing stats
  if (!data.stats || !data.stats.historical || !data.stats.synthetic) {
    return (
      <div className="text-center py-12">
        <p className="text-muted-foreground">Data is loading or unavailable.</p>
      </div>
    )
  }

  const h = data.stats.historical
  const s = data.stats.synthetic

  const fidelityScore = useMemo(() => {
    if (data.overallMatch !== undefined) {
      return Math.round(data.overallMatch * 100)
    }
    // Fallback: compute locally from historical vs synthetic stats
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
  }, [data.overallMatch, h, s])

  const pct = (v: number) => `${(v * 100).toFixed(2)}%`
  const num = (v: number) => v.toFixed(3)
  const rc = (v: number) => (v >= 0 ? "text-emerald-500" : "text-red-500")
  const selectedPathLabel = data.selectedScenarioId
    ? `Path #${data.selectedScenarioId}`
    : "Path #1"
  const objectiveLabel =
    data.selectionObjective === "trend_mean"
      ? "Trend target"
      : data.selectionObjective === "momentum_acf1"
        ? "Momentum target (ACF1)"
        : data.selectionObjective === "fat_tails_extreme_events"
          ? "Fat Tails target (extreme events)"
          : "Default fallback"
  const extremeEventsValue =
    data.selectionObjective === "fat_tails_extreme_events" && data.selectionValue !== undefined
      ? Math.round(data.selectionValue)
      : null
  const extremeEventsTarget =
    data.selectionObjective === "fat_tails_extreme_events" && data.selectionTarget !== undefined
      ? Math.round(data.selectionTarget)
      : null

  return (
    <div className="space-y-8">
      <div className="grid gap-3 grid-cols-2 md:grid-cols-3 lg:grid-cols-5">
        <div className="rounded-xl border border-border bg-card px-5 py-4 col-span-2 md:col-span-1 flex items-center">
          <FidelityGauge score={fidelityScore} />
        </div>
        <div className="rounded-xl border border-border bg-card px-5 py-4 col-span-2 md:col-span-1 lg:col-span-1 flex flex-col justify-center">
          <p className="text-[10px] font-semibold text-muted-foreground uppercase tracking-widest">
            Selected Display Path
          </p>
          <p className="mt-1 text-lg font-bold tabular-nums text-foreground">{selectedPathLabel}</p>
          <p className="text-[11px] text-muted-foreground">{objectiveLabel}</p>
          {extremeEventsValue !== null && (
            <p className="mt-1 text-[11px] text-muted-foreground">
              Extreme events: <span className="font-semibold text-foreground">{extremeEventsValue}</span>
              {extremeEventsTarget !== null ? ` (target ${extremeEventsTarget})` : ""}
            </p>
          )}
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
        </div>
        <div className="grid gap-6 lg:grid-cols-2">
          <div className="rounded-lg border border-blue-500/20 bg-background pt-5 pr-0 pb-5 pl-0">
            <CandlestickChart
              data={data.historical}
              title="Historical"
              height={420}
            />
          </div>
          <div className="rounded-lg border border-amber-500/20 bg-background pt-5 pr-0 pb-5 pl-0">
            <CandlestickChart
              data={data.synthetic}
              title="Synthetic"
              height={420}
            />
          </div>
        </div>
      </section>

      {/* Analytics */}
      <section className="space-y-4">
        <h3 className="text-base font-semibold text-foreground">Analytics</h3>
        <Tabs defaultValue="overlay">
          <TabsList className={`grid w-full ${data.desiredVolatility !== 1.0 ? "max-w-2xl grid-cols-4" : "max-w-md grid-cols-3"}`}>
            <TabsTrigger value="overlay">Price Overlay</TabsTrigger>
            <TabsTrigger value="returns">Cum. Returns</TabsTrigger>
            <TabsTrigger value="drawdown">Drawdowns</TabsTrigger>
            {data.desiredVolatility !== 1.0 && <TabsTrigger value="volatility">Volatility Fan</TabsTrigger>}
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
          {data.desiredVolatility !== 1.0 && (
            <TabsContent value="volatility" className="mt-4">
              <div className="rounded-xl border border-border bg-card p-5">
                {data.volatilityFan && data.volatilityFan.length > 0 ? (
                  <VolatilityFanChart data={data.volatilityFan} height={420} />
                ) : (
                  <p className="text-sm text-muted-foreground">Volatility fan data not available</p>
                )}
              </div>
            </TabsContent>
          )}
        </Tabs>
      </section>

      {/* Statistical comparison */}
      <section>
        <h3 className="mb-4 text-base font-semibold text-foreground">Statistical Comparison</h3>
          <StatsPanel
          stats={{
            ...data.stats,
            historical: {
              ...data.stats.historical,
              numDataPoints: data.historical.length,
            },
            synthetic: {
              ...data.stats.synthetic,
              numDataPoints: data.synthetic.length,
            },
          }}
        />

      </section>
    </div>
  )
}
