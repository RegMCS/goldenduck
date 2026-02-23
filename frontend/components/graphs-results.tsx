"use client"

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

export function VisualizationResults({ data }: VisualizationResultsProps) {
  return (
    <div className="space-y-8">
      {/* Stats overview banner */}
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4 lg:grid-cols-6">
        <StatCard
          label="Ann. Return (H)"
          value={`${(data.stats.historical.annualizedReturn * 100).toFixed(2)}%`}
          variant={data.stats.historical.annualizedReturn >= 0 ? "positive" : "negative"}
        />
        <StatCard
          label="Ann. Return (S)"
          value={`${(data.stats.synthetic.annualizedReturn * 100).toFixed(2)}%`}
          variant={data.stats.synthetic.annualizedReturn >= 0 ? "positive" : "negative"}
        />
        <StatCard
          label="Ann. Vol (H)"
          value={`${(data.stats.historical.annualizedVol * 100).toFixed(2)}%`}
          variant="neutral"
        />
        <StatCard
          label="Ann. Vol (S)"
          value={`${(data.stats.synthetic.annualizedVol * 100).toFixed(2)}%`}
          variant="neutral"
        />
        <StatCard
          label="Sharpe (H)"
          value={data.stats.historical.sharpe.toFixed(4)}
          variant={data.stats.historical.sharpe >= 0 ? "positive" : "negative"}
        />
        <StatCard
          label="Max DD (H)"
          value={`${(data.stats.historical.maxDrawdown * 100).toFixed(2)}%`}
          variant="negative"
        />
      </div>

      {/* Candlestick Charts - Side by Side */}
      <section className="rounded-xl border border-border bg-card p-5">
        <div className="mb-4">
          <h3 className="text-base font-semibold text-foreground">
            OHLCV Candlestick Comparison
          </h3>
          <p className="text-xs text-muted-foreground">
            Side-by-side comparison of historical (left) and synthetic (right) OHLCV data.
            Hover over candles for full OHLC, volume, and change data.
          </p>
        </div>
        <div className="grid gap-4 lg:grid-cols-2">
          <div className="rounded-lg border border-border/50 bg-background p-3">
            <CandlestickChart
              data={data.historical}
              title="Historical Data"
              subtitle={`${data.historical.length} data points | ${data.historical[0]?.date} - ${data.historical[data.historical.length - 1]?.date}`}
              height={360}
            />
          </div>
          <div className="rounded-lg border border-border/50 bg-background p-3">
            <CandlestickChart
              data={data.synthetic}
              title="Synthetic Data"
              subtitle={`${data.synthetic.length} data points | Generated`}
              height={360}
            />
          </div>
        </div>
      </section>

      {/* Tabbed analytical views */}
      <Tabs defaultValue="overlay" className="space-y-4">
        <TabsList className="grid w-full max-w-lg grid-cols-3">
          <TabsTrigger value="overlay">Price Overlay</TabsTrigger>
          <TabsTrigger value="returns">Cumulative Returns</TabsTrigger>
          <TabsTrigger value="drawdown">Drawdowns</TabsTrigger>
        </TabsList>

        <TabsContent value="overlay">
          <div className="grid gap-6 xl:grid-cols-[1fr_340px]">
            <div className="rounded-xl border border-border bg-card p-5">
              <PriceOverlayChart data={data.timeSeries} height={420} />
            </div>
            <StatsPanel stats={data.stats} />
          </div>
        </TabsContent>

        <TabsContent value="returns">
          <div className="grid gap-6 xl:grid-cols-[1fr_340px]">
            <div className="rounded-xl border border-border bg-card p-5">
              <CumulativeReturnChart data={data.returns} height={420} />
            </div>
            <StatsPanel stats={data.stats} />
          </div>
        </TabsContent>

        <TabsContent value="drawdown">
          <div className="grid gap-6 xl:grid-cols-[1fr_340px]">
            <div className="rounded-xl border border-border bg-card p-5">
              <DrawdownChart data={data.drawdowns} height={420} />
            </div>
            <StatsPanel stats={data.stats} />
          </div>
        </TabsContent>
      </Tabs>
    </div>
  )
}

function StatCard({
  label,
  value,
  variant,
}: {
  label: string
  value: string
  variant: "positive" | "negative" | "neutral"
}) {
  return (
    <div className="rounded-lg border border-border bg-card p-3">
      <p className="text-xs text-muted-foreground truncate">{label}</p>
      <p
        className={`mt-1 text-sm font-mono font-semibold tabular-nums ${
          variant === "positive"
            ? "text-emerald-500"
            : variant === "negative"
              ? "text-red-500"
              : "text-foreground"
        }`}
      >
        {value}
      </p>
    </div>
  )
}
