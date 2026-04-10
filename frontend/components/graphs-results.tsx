"use client"

import { useMemo } from "react"
import { type GeneratedData } from "@/lib/types"
import { CandlestickChart } from "@/components/charts/candlestick-chart"
import { PriceOverlayChart } from "@/components/charts/price-overlay-chart"
import { CumulativeReturnChart } from "@/components/charts/cumulative-return-chart"
import { DrawdownChart } from "@/components/charts/drawdown-chart"
import { VolatilityFanChart } from "@/components/charts/volatility-fan-chart"
import { DistributionOverlayChart } from "@/components/charts/distribution-overlay-chart"
import { StatsPanel } from "@/components/charts/stats-panel"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { InfoTooltip } from "@/components/info-tooltip"

interface VisualizationResultsProps {
  data: GeneratedData
}

function quantile(values: number[], q: number): number {
  if (values.length === 0) return 0
  const sorted = [...values].sort((a, b) => a - b)
  const pos = (sorted.length - 1) * q
  const base = Math.floor(pos)
  const rest = pos - base
  const next = sorted[base + 1] ?? sorted[base]
  return sorted[base] + rest * (next - sorted[base])
}

function klDivergenceFromReturns(pSamples: number[], qSamples: number[], bins = 40): number {
  if (pSamples.length < 20 || qSamples.length < 20) return 0

  const all = [...pSamples, ...qSamples].filter((v) => Number.isFinite(v))
  if (all.length < 40) return 0

  const minV = Math.min(...all)
  const maxV = Math.max(...all)
  if (!Number.isFinite(minV) || !Number.isFinite(maxV) || maxV <= minV) return 0

  const pCounts = new Array(bins).fill(0)
  const qCounts = new Array(bins).fill(0)
  const width = (maxV - minV) / bins

  const toBin = (v: number) => {
    if (v <= minV) return 0
    if (v >= maxV) return bins - 1
    return Math.min(bins - 1, Math.max(0, Math.floor((v - minV) / width)))
  }

  for (const v of pSamples) pCounts[toBin(v)] += 1
  for (const v of qSamples) qCounts[toBin(v)] += 1

  // Additive smoothing to avoid log(0)
  const eps = 1e-8
  const pDen = pSamples.length + eps * bins
  const qDen = qSamples.length + eps * bins

  let kl = 0
  for (let i = 0; i < bins; i++) {
    const p = (pCounts[i] + eps) / pDen
    const q = (qCounts[i] + eps) / qDen
    kl += p * Math.log(p / q)
  }

  if (!Number.isFinite(kl)) return 0
  return Math.max(0, kl)
}

function hurstMomentum(values: number[]): number {
  if (values.length < 20) return 0.5

  const mean = values.reduce((a, b) => a + b, 0) / values.length
  const y: number[] = new Array(values.length)
  let csum = 0
  for (let i = 0; i < values.length; i++) {
    csum += values[i] - mean
    y[i] = csum
  }

  const maxLag = Math.floor(y.length / 2)
  const baseLags = [2, 4, 8, 16, 32, 64]
  const lags = baseLags.filter((lag) => lag < maxLag)
  if (lags.length < 2) return 0.5

  const logLags: number[] = []
  const logTau: number[] = []
  for (const lag of lags) {
    const diffs: number[] = []
    for (let i = lag; i < y.length; i++) {
      diffs.push(y[i] - y[i - lag])
    }
    if (diffs.length < 2) continue
    const dMean = diffs.reduce((a, b) => a + b, 0) / diffs.length
    const dVar = diffs.reduce((a, b) => a + (b - dMean) * (b - dMean), 0) / (diffs.length - 1)
    const tau = Math.sqrt(Math.max(dVar, 0))
    if (tau > 1e-12 && Number.isFinite(tau)) {
      logLags.push(Math.log(lag))
      logTau.push(Math.log(tau))
    }
  }

  if (logLags.length < 2) return 0.5

  const xMean = logLags.reduce((a, b) => a + b, 0) / logLags.length
  const yMean = logTau.reduce((a, b) => a + b, 0) / logTau.length
  let num = 0
  let den = 0
  for (let i = 0; i < logLags.length; i++) {
    const dx = logLags[i] - xMean
    num += dx * (logTau[i] - yMean)
    den += dx * dx
  }
  if (den <= 1e-12) return 0.5

  const hurst = num / den
  if (!Number.isFinite(hurst)) return 0.5
  return Math.max(0, Math.min(1, hurst))
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

  const enrichedStats = useMemo(() => {
    const hReturns = data.returns.map((r) => r.historicalReturn).filter((v) => Number.isFinite(v))
    const sReturns = data.returns.map((r) => r.syntheticReturn).filter((v) => Number.isFinite(v))

    const hVar95 = Math.max(0, -quantile(hReturns, 0.05))
    const sVar95 = Math.max(0, -quantile(sReturns, 0.05))
    const hHurstMomentum = hurstMomentum(hReturns)
    const sHurstMomentum = hurstMomentum(sReturns)
    const hLegacyMappedMomentum =
      typeof data.stats.historical.acfLag1 === "number"
        ? Math.max(0, Math.min(1, 0.5 + 0.5 * data.stats.historical.acfLag1))
        : hHurstMomentum
    const sLegacyMappedMomentum =
      typeof data.stats.synthetic.acfLag1 === "number"
        ? Math.max(0, Math.min(1, 0.5 + 0.5 * data.stats.synthetic.acfLag1))
        : sHurstMomentum
    const shouldShowKLDivergence = data.selectionObjective === "baseline_composite_match"
    const klDivergence = shouldShowKLDivergence
      ? klDivergenceFromReturns(hReturns, sReturns)
      : undefined

    return {
      ...data.stats,
      historical: {
        ...data.stats.historical,
        var95: data.stats.historical.var95 ?? hVar95,
        hurstMomentum: data.stats.historical.hurstMomentum ?? hLegacyMappedMomentum,
        numDataPoints: data.historical.length,
      },
      synthetic: {
        ...data.stats.synthetic,
        var95: data.stats.synthetic.var95 ?? sVar95,
        hurstMomentum: data.stats.synthetic.hurstMomentum ?? sLegacyMappedMomentum,
        klDivergence:
          typeof data.stats.synthetic.klDivergence === "number"
            ? data.stats.synthetic.klDivergence
            : klDivergence,
        numDataPoints: data.synthetic.length,
      },
    }
  }, [
    data.stats,
    data.returns,
    data.historical.length,
    data.synthetic.length,
    data.selectionObjective,
  ])


  const pct = (v: number) => `${(v * 100).toFixed(2)}%`
  const num = (v: number) => v.toFixed(3)
  const rc = (v: number) => (v >= 0 ? "text-emerald-500" : "text-red-500")
  const selectedPathLabel = data.selectedScenarioId
    ? `Path #${data.selectedScenarioId}`
    : "Path #1"
  const objectiveLabel =
    data.selectionObjective === "trend_mean"
      ? "Trend target"
      : data.selectionObjective === "momentum_hurst"
        ? "Momentum target (Hurst)"
        : data.selectionObjective === "momentum_acf1"
          ? "Momentum target (legacy ACF1)"
        : data.selectionObjective === "fat_tails_extreme_events"
          ? "Fat Tails target (extreme events)"
        : data.selectionObjective === "fat_tails_kurtosis"
          ? "Fat Tails target (kurtosis)"
          : data.selectionObjective === "volatility_std"
            ? "Volatility target (std scaling)"
            : data.selectionObjective === "baseline_composite_match"
              ? "Default mode (best input match)"
          : data.selectionObjective === "volatility_tortuosity"
            ? "Volatility target (path roughness)"
          : data.selectionObjective === "bull_run_composite"
            ? "Bull Run preset (composite score)"
            : data.selectionObjective === "flash_crash_composite"
              ? "Flash Crash preset (composite score)"
          : "Default fallback"
  const extremeEventsValue =
    data.selectionObjective === "fat_tails_extreme_events" && data.selectionValue !== undefined
      ? Math.round(data.selectionValue)
      : null
  const extremeEventsTarget =
    data.selectionObjective === "fat_tails_extreme_events" && data.selectionTarget !== undefined
      ? Math.round(data.selectionTarget)
      : null
  const fatKurtosisValue =
    data.selectionObjective === "fat_tails_kurtosis" && typeof data.selectionValue === "number" && Number.isFinite(data.selectionValue)
      ? data.selectionValue.toFixed(3)
      : null
  const fatKurtosisTarget =
    data.selectionObjective === "fat_tails_kurtosis" && typeof data.selectionTarget === "number" && Number.isFinite(data.selectionTarget)
      ? data.selectionTarget.toFixed(3)
      : null
  const volatilityRatioValue =
    data.selectionObjective === "volatility_tortuosity" && typeof data.selectionValue === "number" && Number.isFinite(data.selectionValue)
      ? data.selectionValue.toFixed(2)
      : null
  const volatilityRatioTarget =
    data.selectionObjective === "volatility_tortuosity" && typeof data.selectionTarget === "number" && Number.isFinite(data.selectionTarget)
      ? data.selectionTarget.toFixed(2)
      : null
  const volatilityStdValue =
    data.selectionObjective === "volatility_std" && typeof data.selectionValue === "number" && Number.isFinite(data.selectionValue)
      ? `${(data.selectionValue * Math.sqrt(252) * 100).toFixed(2)}%`
      : null
  const volatilityStdTarget =
    data.selectionObjective === "volatility_std" && typeof data.selectionTarget === "number" && Number.isFinite(data.selectionTarget)
      ? `${(data.selectionTarget * Math.sqrt(252) * 100).toFixed(2)}%`
      : null
  const presetCompositeScore =
    (data.selectionObjective === "bull_run_composite" ||
      data.selectionObjective === "flash_crash_composite") &&
    typeof data.selectionValue === "number" &&
    Number.isFinite(data.selectionValue)
      ? data.selectionValue.toFixed(3)
      : null
  const baselineCompositeDistance =
    data.selectionObjective === "baseline_composite_match" &&
    typeof data.selectionValue === "number" &&
    Number.isFinite(data.selectionValue)
      ? data.selectionValue.toFixed(3)
      : null
  const weightedSelectionScore =
    typeof data.selectionScore === "number" && Number.isFinite(data.selectionScore)
      ? data.selectionScore.toFixed(3)
      : baselineCompositeDistance
  const isWeightedObjective =
    data.selectionObjective === "baseline_composite_match" ||
    data.selectionObjective === "volatility_std" ||
    data.selectionObjective === "trend_mean" ||
    data.selectionObjective === "momentum_hurst" ||
    data.selectionObjective === "fat_tails_extreme_events" ||
    data.selectionObjective === "fat_tails_kurtosis"
  const showVolatilityTab = data.desiredVolatility !== 1.0
  const showFatTailsDistributionTab =
    data.selectionObjective === "fat_tails_kurtosis" ||
    data.selectionObjective === "fat_tails_extreme_events"
  const analyticsCols = showVolatilityTab && showFatTailsDistributionTab
    ? "max-w-3xl grid-cols-5"
    : showVolatilityTab || showFatTailsDistributionTab
      ? "max-w-2xl grid-cols-4"
      : "max-w-md grid-cols-3"

  return (
    <div className="space-y-3">
      {/* Row: Path info + all metric cards */}
      <div className="grid gap-3 grid-cols-2 sm:grid-cols-3 lg:grid-cols-6">
        <div className="rounded-xl border border-border bg-card p-4 space-y-3 min-w-0">
          <div className="flex items-center gap-1">
            <p className="text-[10px] font-semibold text-muted-foreground uppercase tracking-widest truncate">
              Display Path
            </p>
            <InfoTooltip content={[
              `Selection mode: ${objectiveLabel}`,
              ...(extremeEventsValue !== null ? [`Extreme events: ${extremeEventsValue}${extremeEventsTarget !== null ? ` (target ${extremeEventsTarget})` : ""}`] : []),
              ...(fatKurtosisValue !== null ? [`Excess kurtosis: ${fatKurtosisValue}${fatKurtosisTarget !== null ? ` (target ${fatKurtosisTarget})` : ""}`] : []),
              ...(volatilityRatioValue !== null ? [`Roughness ratio: ${volatilityRatioValue}${volatilityRatioTarget !== null ? ` (target ${volatilityRatioTarget})` : ""}`] : []),
              ...(volatilityStdValue !== null ? [`Annualised vol: ${volatilityStdValue}${volatilityStdTarget !== null ? ` (target ${volatilityStdTarget})` : ""}`] : []),
              ...(presetCompositeScore !== null ? [`Composite score: ${presetCompositeScore}`] : []),
              ...(isWeightedObjective && weightedSelectionScore !== null ? [`Weighted score: ${weightedSelectionScore} — lower means a closer match to your inputs`] : []),
            ]} />
          </div>
          <p className="text-sm font-mono font-bold tabular-nums text-foreground">{selectedPathLabel}</p>
          <p className="text-[10px] text-muted-foreground truncate">{objectiveLabel}</p>
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
        <DualMetricCard
          label="Total Return"
          histValue={pct(h.totalReturn)}
          synthValue={pct(s.totalReturn)}
          histClass={rc(h.totalReturn)}
          synthClass={rc(s.totalReturn)}
          delta={`${((s.totalReturn - h.totalReturn) * 100).toFixed(2)}pp`}
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
          <TabsList className={`grid w-full ${analyticsCols}`}>
            <TabsTrigger value="overlay">Price Overlay</TabsTrigger>
            <TabsTrigger value="returns">Cum. Returns</TabsTrigger>
            <TabsTrigger value="drawdown">Drawdowns</TabsTrigger>
            {showVolatilityTab && <TabsTrigger value="volatility">Volatility Fan</TabsTrigger>}
            {showFatTailsDistributionTab && <TabsTrigger value="distribution">Distribution</TabsTrigger>}
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
          {showVolatilityTab && (
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
          {showFatTailsDistributionTab && (
            <TabsContent value="distribution" className="mt-4">
              <div className="rounded-xl border border-border bg-card p-5">
                <DistributionOverlayChart
                  data={data.returns}
                  historicalKurtosis={data.stats?.historical?.kurtosis}
                  syntheticKurtosis={data.stats?.synthetic?.kurtosis}
                  height={420}
                />
              </div>
            </TabsContent>
          )}
        </Tabs>
      </section>

      {/* Statistical comparison */}
      <section>
        <h3 className="mb-4 text-base font-semibold text-foreground">Statistical Comparison</h3>
          <StatsPanel
          stats={enrichedStats}
        />

      </section>
    </div>
  )
}
