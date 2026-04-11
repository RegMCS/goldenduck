"use client"

import { useCallback, useEffect, useState } from "react"
import {
  RefreshCw,
  CheckCircle2,
  XCircle,
  Loader2,
  TrendingUp,
  Target,
  BarChart3,
  Activity,
  Star,
} from "lucide-react"
import { Button } from "@/components/ui/button"
import { type EvaluationReport, type DirectMetrics, type BaselineComparison } from "@/lib/types"
import { fetchTrainingGet } from "@/lib/training-fetch"

// ── Metric helpers ────────────────────────────────────────────────────────────

function fmt(v: number, decimals = 4) {
  return v.toFixed(decimals)
}

function PassBadge({ pass }: { pass: boolean }) {
  return pass ? (
    <span className="inline-flex items-center gap-1 rounded-full border border-emerald-200 bg-emerald-100 px-2 py-0.5 text-xs font-semibold text-emerald-700 dark:border-emerald-700 dark:bg-emerald-900/30 dark:text-emerald-400">
      <CheckCircle2 className="h-3 w-3" /> PASS
    </span>
  ) : (
    <span className="inline-flex items-center gap-1 rounded-full border border-red-200 bg-red-100 px-2 py-0.5 text-xs font-semibold text-red-600 dark:border-red-700 dark:bg-red-900/30 dark:text-red-400">
      <XCircle className="h-3 w-3" /> FAIL
    </span>
  )
}

// Horizontal gauge bar (0 → 1 scale)
function Gauge({ value, thresholds }: { value: number; thresholds: { good: number; ok: number } }) {
  const pct = Math.max(0, Math.min(100, value * 100))
  const color =
    value >= thresholds.good
      ? "bg-emerald-500"
      : value >= thresholds.ok
      ? "bg-amber-500"
      : "bg-red-500"

  return (
    <div className="space-y-1.5">
      <div className="flex items-center justify-between text-xs text-muted-foreground">
        <span>0.0</span>
        <span className="font-bold text-foreground text-sm">{fmt(value, 3)}</span>
        <span>1.0</span>
      </div>
      <div className="h-3 w-full rounded-full bg-muted overflow-hidden">
        <div
          className={`h-full rounded-full transition-all duration-700 ${color}`}
          style={{ width: `${pct}%` }}
        />
      </div>
      {/* Threshold markers */}
      <div className="relative h-1">
        <div
          className="absolute top-0 h-2 w-0.5 rounded-full bg-amber-400"
          style={{ left: `${thresholds.ok * 100}%` }}
          title={`Acceptable: ${thresholds.ok}`}
        />
        <div
          className="absolute top-0 h-2 w-0.5 rounded-full bg-emerald-500"
          style={{ left: `${thresholds.good * 100}%` }}
          title={`Good: ${thresholds.good}`}
        />
      </div>
    </div>
  )
}

// Mini horizontal bar chart for baseline comparison
function BaselineBar({ label, value, max, highlight }: { label: string; value: number; max: number; highlight?: boolean }) {
  const pct = max > 0 ? (value / max) * 100 : 0
  return (
    <div className="flex items-center gap-3">
      <span className={`text-xs w-24 flex-shrink-0 ${highlight ? "font-semibold text-foreground" : "text-muted-foreground"}`}>
        {label}
      </span>
      <div className="flex-1 h-2 rounded-full bg-muted overflow-hidden">
        <div
          className={`h-full rounded-full transition-all duration-700 ${highlight ? "bg-primary" : "bg-muted-foreground/40"}`}
          style={{ width: `${pct}%` }}
        />
      </div>
      <span className={`text-xs tabular-nums w-12 text-right ${highlight ? "font-bold text-primary" : "text-muted-foreground"}`}>
        {fmt(value, 3)}
      </span>
    </div>
  )
}

function DirectMetricsCard({ metrics }: { metrics: DirectMetrics }) {
  const rows = [
    { label: "R² Score", value: fmt(metrics.r2, 4), pass: metrics.r2 >= 0.7, note: "≥ 0.7 good" },
    { label: "RMSE", value: fmt(metrics.rmse, 4), pass: metrics.rmse <= 0.3, note: "≤ 0.30 good" },
    { label: "MAE", value: fmt(metrics.mae, 4), pass: metrics.mae <= 0.25, note: "≤ 0.25 good" },
    ...(metrics.mape !== undefined
      ? [{ label: "MAPE", value: fmt(metrics.mape, 2) + "%", pass: metrics.mape <= 30, note: "≤ 30% good" }]
      : []),
  ]

  return (
    <div className="rounded-2xl border border-border bg-card shadow-sm overflow-hidden">
      <div className="flex items-center gap-2.5 border-b border-border bg-muted/30 px-5 py-3.5">
        <div className="flex h-7 w-7 items-center justify-center rounded-lg bg-primary/10">
          <Target className="h-4 w-4 text-primary" />
        </div>
        <div>
          <h4 className="text-sm font-semibold text-foreground">Delta Parameter — Direct Metrics</h4>
          <p className="text-xs text-muted-foreground">How accurately the RF model predicts δ on the test set</p>
        </div>
      </div>
      <div className="p-5">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-border">
              <th className="pb-2 text-left text-xs font-semibold uppercase tracking-wider text-muted-foreground">Metric</th>
              <th className="pb-2 text-right text-xs font-semibold uppercase tracking-wider text-muted-foreground">Value</th>
              <th className="pb-2 text-right text-xs font-semibold uppercase tracking-wider text-muted-foreground">Threshold</th>
              <th className="pb-2 text-right text-xs font-semibold uppercase tracking-wider text-muted-foreground">Result</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-border">
            {rows.map((r) => (
              <tr key={r.label}>
                <td className="py-2.5 text-foreground font-medium">{r.label}</td>
                <td className="py-2.5 text-right font-mono tabular-nums text-foreground">{r.value}</td>
                <td className="py-2.5 text-right text-xs text-muted-foreground">{r.note}</td>
                <td className="py-2.5 text-right">
                  <PassBadge pass={r.pass} />
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}

function EndToEndCard({ score }: { score: number }) {
  return (
    <div className="rounded-2xl border border-border bg-card shadow-sm overflow-hidden">
      <div className="flex items-center gap-2.5 border-b border-border bg-muted/30 px-5 py-3.5">
        <div className="flex h-7 w-7 items-center justify-center rounded-lg bg-primary/10">
          <Activity className="h-4 w-4 text-primary" />
        </div>
        <div>
          <h4 className="text-sm font-semibold text-foreground">End-to-End Synthetic Quality</h4>
          <p className="text-xs text-muted-foreground">How well AI-generated scenarios match user objectives (0–1 scale)</p>
        </div>
      </div>
      <div className="p-5 space-y-4">
        <Gauge value={score} thresholds={{ good: 0.7, ok: 0.5 }} />
        <div className="grid grid-cols-3 gap-3 text-center text-xs">
          <div className="rounded-lg bg-red-50 dark:bg-red-900/20 px-2 py-2">
            <p className="text-red-600 dark:text-red-400 font-semibold">Poor</p>
            <p className="text-muted-foreground mt-0.5">&lt; 0.5</p>
          </div>
          <div className="rounded-lg bg-amber-50 dark:bg-amber-900/20 px-2 py-2">
            <p className="text-amber-600 dark:text-amber-400 font-semibold">Acceptable</p>
            <p className="text-muted-foreground mt-0.5">0.5 – 0.7</p>
          </div>
          <div className="rounded-lg bg-emerald-50 dark:bg-emerald-900/20 px-2 py-2">
            <p className="text-emerald-600 dark:text-emerald-400 font-semibold">Good</p>
            <p className="text-muted-foreground mt-0.5">&gt; 0.7</p>
          </div>
        </div>
        <div className="flex justify-center">
          <PassBadge pass={score >= 0.7} />
        </div>
      </div>
    </div>
  )
}

function BaselineCard({ baseline }: { baseline: BaselineComparison }) {
  const max = Math.max(baseline.ai_score, baseline.baseline_fixed, baseline.baseline_heuristic) * 1.1

  return (
    <div className="rounded-2xl border border-border bg-card shadow-sm overflow-hidden">
      <div className="flex items-center gap-2.5 border-b border-border bg-muted/30 px-5 py-3.5">
        <div className="flex h-7 w-7 items-center justify-center rounded-lg bg-primary/10">
          <BarChart3 className="h-4 w-4 text-primary" />
        </div>
        <div>
          <h4 className="text-sm font-semibold text-foreground">Baseline Comparison</h4>
          <p className="text-xs text-muted-foreground">AI vs fixed parameters vs simple heuristic</p>
        </div>
      </div>
      <div className="p-5 space-y-3">
        <BaselineBar label="Random Forest" value={baseline.ai_score} max={max} highlight />
        <BaselineBar label="Fixed Params" value={baseline.baseline_fixed} max={max} />
        <BaselineBar label="Heuristic" value={baseline.baseline_heuristic} max={max} />

        <div className="mt-4 grid grid-cols-2 gap-3 text-xs">
          <div className="rounded-lg border border-border bg-background px-3 py-2">
            <p className="text-muted-foreground">vs Fixed p-value</p>
            <p className={`mt-0.5 font-mono font-semibold ${baseline.p_value_vs_fixed < 0.05 ? "text-emerald-600 dark:text-emerald-400" : "text-red-500"}`}>
              {baseline.p_value_vs_fixed.toExponential(2)}
            </p>
            <PassBadge pass={baseline.p_value_vs_fixed < 0.05} />
          </div>
          <div className="rounded-lg border border-border bg-background px-3 py-2">
            <p className="text-muted-foreground">vs Heuristic p-value</p>
            <p className={`mt-0.5 font-mono font-semibold ${baseline.p_value_vs_heuristic < 0.05 ? "text-emerald-600 dark:text-emerald-400" : "text-amber-500"}`}>
              {baseline.p_value_vs_heuristic.toExponential(2)}
            </p>
            <PassBadge pass={baseline.p_value_vs_heuristic < 0.05} />
          </div>
        </div>
      </div>
    </div>
  )
}

function PassFailChecklist({ checks, overall }: { checks: Record<string, boolean>; overall: boolean }) {
  return (
    <div className="rounded-2xl border border-border bg-card shadow-sm overflow-hidden">
      <div className="flex items-center gap-2.5 border-b border-border bg-muted/30 px-5 py-3.5">
        <div className="flex h-7 w-7 items-center justify-center rounded-lg bg-primary/10">
          <TrendingUp className="h-4 w-4 text-primary" />
        </div>
        <div>
          <h4 className="text-sm font-semibold text-foreground">Production Readiness</h4>
          <p className="text-xs text-muted-foreground">Pass all checks to deploy this model</p>
        </div>
        <div className="ml-auto">
          <PassBadge pass={overall} />
        </div>
      </div>
      <div className="p-5 space-y-2">
        {Object.entries(checks).map(([criterion, passed]) => (
          <div
            key={criterion}
            className={`flex items-center gap-3 rounded-xl px-4 py-2.5 ${
              passed ? "bg-emerald-50 dark:bg-emerald-900/20" : "bg-red-50 dark:bg-red-900/20"
            }`}
          >
            {passed ? (
              <CheckCircle2 className="h-4 w-4 text-emerald-500 flex-shrink-0" />
            ) : (
              <XCircle className="h-4 w-4 text-red-500 flex-shrink-0" />
            )}
            <span className={`text-sm ${passed ? "text-emerald-700 dark:text-emerald-400" : "text-red-700 dark:text-red-400"}`}>
              {criterion}
            </span>
          </div>
        ))}
      </div>
    </div>
  )
}

interface Props {
  /** If provided, display this specific report (for the active model). */
  report?: EvaluationReport | null
}

export function EvaluationReportCard({ report: propReport }: Props) {
  const [report, setReport] = useState<EvaluationReport | null>(propReport ?? null)
  const [loading, setLoading] = useState(!propReport)
  const [error, setError] = useState<string | null>(null)

  const fetchActiveReport = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const res = await fetchTrainingGet("/api/training/active-model/evaluation")
      if (!res.ok) {
        let msg = `Server returned ${res.status}`
        try {
          const body = await res.json()
          if (body?.detail && typeof body.detail === "string") msg = body.detail
        } catch {
          /* ignore */
        }
        throw new Error(msg)
      }
      const data = await res.json()
      setReport(data)
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unknown error")
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    if (!propReport) {
      fetchActiveReport()
    }
  }, [propReport, fetchActiveReport])

  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center py-20 gap-3">
        <Loader2 className="h-8 w-8 animate-spin text-primary" />
        <p className="text-sm text-muted-foreground">Loading evaluation report…</p>
      </div>
    )
  }

  if (error || !report) {
    return (
      <div className="flex flex-col items-center justify-center py-20 gap-3 text-center">
        <div className="flex h-14 w-14 items-center justify-center rounded-2xl bg-muted">
          <BarChart3 className="h-7 w-7 text-muted-foreground" />
        </div>
        <h3 className="text-sm font-semibold text-foreground">No evaluation report available</h3>
        <p className="text-xs text-muted-foreground max-w-xs">
          {error ?? "Train a model and evaluation will run automatically."}
        </p>
        {error && (
          <Button variant="outline" size="sm" onClick={fetchActiveReport} className="gap-1.5">
            <RefreshCw className="h-3.5 w-3.5" />
            Retry
          </Button>
        )}
      </div>
    )
  }

  return (
    <div className="space-y-5">
      {/* Header strip */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Star className="h-4 w-4 text-amber-500 fill-amber-500" />
          <p className="text-sm font-semibold text-foreground">Active Model Evaluation</p>
          <span className="text-xs text-muted-foreground font-mono">
            {new Date(report.timestamp).toLocaleString("en-SG")}
          </span>
        </div>
        <Button
          variant="outline"
          size="sm"
          className="gap-1.5 h-8"
          onClick={fetchActiveReport}
          id="refresh-eval-btn"
        >
          <RefreshCw className="h-3.5 w-3.5" />
          Refresh
        </Button>
      </div>

      {/* Dataset sizes */}
      <div className="grid grid-cols-3 gap-3">
        {(["train", "val", "test"] as const).map((split) => (
          <div key={split} className="rounded-xl border border-border bg-card px-4 py-3 text-center shadow-sm">
            <p className="text-xs capitalize text-muted-foreground">{split} samples</p>
            <p className="text-2xl font-bold tabular-nums text-foreground mt-0.5">
              {report.dataset_size[split].toLocaleString()}
            </p>
          </div>
        ))}
      </div>

      {/* Cards */}
      <DirectMetricsCard metrics={report.direct_metrics.delta} />
      <EndToEndCard score={report.end_to_end_metrics.mean_score} />
      <BaselineCard baseline={report.baseline_comparison} />
      <PassFailChecklist checks={report.quality_checks} overall={report.overall_pass} />
    </div>
  )
}
