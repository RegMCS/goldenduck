"use client"

import { useCallback, useEffect, useState } from "react"
import {
  RefreshCw,
  CheckCircle2,
  XCircle,
  Loader2,
  Clock,
  Calendar,
  Zap,
  ChevronDown,
  ChevronUp,
  BarChart3,
  Star,
} from "lucide-react"
import { Button } from "@/components/ui/button"
import { type TrainingRun, type TrainingStatus, type EvaluationReport } from "@/lib/types"

function formatDate(iso: string) {
  return new Intl.DateTimeFormat("en-SG", {
    dateStyle: "medium",
    timeStyle: "short",
  }).format(new Date(iso))
}

function duration(start: string, end: string | null) {
  if (!end) return "—"
  const ms = new Date(end).getTime() - new Date(start).getTime()
  const secs = Math.floor(ms / 1000)
  if (secs < 60) return `${secs}s`
  const mins = Math.floor(secs / 60)
  if (mins < 60) return `${mins}m ${secs % 60}s`
  const hours = Math.floor(mins / 60)
  return `${hours}h ${mins % 60}m`
}

function StatusBadge({ status }: { status: TrainingStatus }) {
  const map: Record<TrainingStatus, { label: string; icon: React.ReactNode; className: string }> = {
    queued: {
      label: "Queued",
      icon: <Clock className="h-3 w-3" />,
      className: "bg-amber-100 text-amber-700 border-amber-200 dark:bg-amber-900/30 dark:text-amber-400 dark:border-amber-700",
    },
    running: {
      label: "Running",
      icon: <Loader2 className="h-3 w-3 animate-spin" />,
      className: "bg-blue-100 text-blue-700 border-blue-200 dark:bg-blue-900/30 dark:text-blue-400 dark:border-blue-700",
    },
    completed: {
      label: "Completed",
      icon: <CheckCircle2 className="h-3 w-3" />,
      className: "bg-emerald-100 text-emerald-700 border-emerald-200 dark:bg-emerald-900/30 dark:text-emerald-400 dark:border-emerald-700",
    },
    failed: {
      label: "Failed",
      icon: <XCircle className="h-3 w-3" />,
      className: "bg-red-100 text-red-700 border-red-200 dark:bg-red-900/30 dark:text-red-400 dark:border-red-700",
    },
  }
  const { label, icon, className } = map[status] ?? map.queued
  return (
    <span className={`inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-xs font-medium ${className}`}>
      {icon}
      {label}
    </span>
  )
}

function R2Indicator({ value }: { value: number }) {
  const pct = Math.max(0, Math.min(100, value * 100))
  const color = value >= 0.8 ? "bg-emerald-500" : value >= 0.6 ? "bg-amber-500" : "bg-red-500"
  return (
    <div className="flex items-center gap-2 min-w-0">
      <div className="flex-1 h-1.5 rounded-full bg-muted overflow-hidden">
        <div className={`h-full rounded-full transition-all ${color}`} style={{ width: `${pct}%` }} />
      </div>
      <span className={`text-xs font-semibold tabular-nums ${value >= 0.7 ? "text-emerald-600 dark:text-emerald-400" : value >= 0.5 ? "text-amber-600 dark:text-amber-400" : "text-red-600 dark:text-red-400"}`}>
        {value >= 0 ? value.toFixed(2) : value.toFixed(2)}
      </span>
    </div>
  )
}

function EvalSummaryRow({ report }: { report: EvaluationReport }) {
  const delta = report.direct_metrics.delta
  const e2e = report.end_to_end_metrics.mean_score
  return (
    <div className="grid grid-cols-3 gap-3 mt-3">
      <div className="rounded-lg border border-border bg-background px-3 py-2">
        <p className="text-xs text-muted-foreground">Delta R²</p>
        <R2Indicator value={delta.r2} />
      </div>
      <div className="rounded-lg border border-border bg-background px-3 py-2">
        <p className="text-xs text-muted-foreground">End-to-End</p>
        <R2Indicator value={e2e} />
      </div>
      <div className="rounded-lg border border-border bg-background px-3 py-2">
        <p className="text-xs text-muted-foreground">Overall</p>
        <div className="mt-0.5">
          {report.overall_pass ? (
            <span className="text-xs font-semibold text-emerald-600 dark:text-emerald-400">✓ Pass</span>
          ) : (
            <span className="text-xs font-semibold text-red-500">✗ Fail</span>
          )}
        </div>
      </div>
    </div>
  )
}

function SkeletonRow() {
  return (
    <tr className="border-b border-border animate-pulse">
      {Array.from({ length: 6 }).map((_, i) => (
        <td key={i} className="px-4 py-4">
          <div className="h-4 w-full rounded-md bg-muted" />
        </td>
      ))}
    </tr>
  )
}

interface Props {
  refreshTrigger: number
}

export function TrainingRunTable({ refreshTrigger }: Props) {
  const [runs, setRuns] = useState<TrainingRun[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [expandedRow, setExpandedRow] = useState<string | null>(null)
  const [activatingId, setActivatingId] = useState<string | null>(null)

  const fetchRuns = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const res = await fetch("/api/training/runs")
      if (!res.ok) throw new Error(`Server returned ${res.status}`)
      const data = await res.json()
      setRuns(data.runs ?? [])
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unknown error")
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    fetchRuns()
  }, [fetchRuns, refreshTrigger])

  async function handleActivate(run: TrainingRun) {
    if (!run.model_name) return
    setActivatingId(run.id)
    try {
      const res = await fetch(`/api/training/models/${run.model_name}/activate`, {
        method: "POST",
      })
      if (!res.ok) throw new Error("Failed to activate model")
      await fetchRuns()
    } catch (err) {
      alert(err instanceof Error ? err.message : "Failed to activate")
    } finally {
      setActivatingId(null)
    }
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <p className="text-sm text-muted-foreground">{runs.length} training run{runs.length !== 1 ? "s" : ""} total</p>
        <Button
          variant="outline"
          size="sm"
          className="gap-1.5 h-8"
          onClick={fetchRuns}
          disabled={loading}
          id="refresh-runs-btn"
        >
          <RefreshCw className={`h-3.5 w-3.5 ${loading ? "animate-spin" : ""}`} />
          Refresh
        </Button>
      </div>

      <div className="rounded-2xl border border-border bg-card shadow-sm overflow-hidden">
        {error && (
          <div className="flex items-center gap-2 border-b border-border bg-destructive/10 px-6 py-3 text-sm text-destructive">
            <XCircle className="h-4 w-4 flex-shrink-0" />
            Failed to load runs: {error}
          </div>
        )}

        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-border bg-muted/40">
                {["Run ID", "Status", "Config", "Started", "Duration", "Delta R²", "Actions"].map((h) => (
                  <th
                    key={h}
                    className="px-4 py-3 text-left text-xs font-semibold uppercase tracking-wider text-muted-foreground"
                  >
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {loading && Array.from({ length: 5 }).map((_, i) => <SkeletonRow key={i} />)}

              {!loading && runs.length === 0 && (
                <tr>
                  <td colSpan={7}>
                    <div className="flex flex-col items-center justify-center py-20 text-center">
                      <div className="mb-4 flex h-14 w-14 items-center justify-center rounded-2xl bg-muted">
                        <BarChart3 className="h-7 w-7 text-muted-foreground" />
                      </div>
                      <h3 className="text-sm font-semibold text-foreground">No training runs yet</h3>
                      <p className="mt-1 text-xs text-muted-foreground">
                        Go to "Trigger Training" to start your first run.
                      </p>
                    </div>
                  </td>
                </tr>
              )}

              {!loading &&
                runs.map((run) => {
                  const isExpanded = expandedRow === run.id
                  const deltaR2 = run.evaluation_report?.direct_metrics?.delta?.r2

                  return (
                    <>
                      <tr
                        key={run.id}
                        className="border-b border-border/60 transition-colors hover:bg-muted/20 last:border-0"
                      >
                        {/* Run ID */}
                        <td className="px-4 py-3.5 font-mono text-xs text-muted-foreground">
                          <div className="flex items-center gap-1.5">
                            {run.is_active && (
                              <Star className="h-3 w-3 text-amber-500 fill-amber-500 flex-shrink-0" />
                            )}
                            <span title={run.id}>{run.id.slice(0, 8)}…</span>
                          </div>
                        </td>

                        {/* Status */}
                        <td className="px-4 py-3.5">
                          <StatusBadge status={run.status} />
                        </td>

                        {/* Config */}
                        <td className="px-4 py-3.5 text-xs text-muted-foreground whitespace-nowrap">
                          {run.config.n_assets}A × {run.config.n_scenarios}S
                          {run.config.testing_mode && (
                            <span className="ml-1.5 rounded-sm bg-amber-100 dark:bg-amber-900/30 text-amber-700 dark:text-amber-400 px-1 py-0.5 text-[10px] font-medium">
                              TEST
                            </span>
                          )}
                        </td>

                        {/* Started */}
                        <td className="px-4 py-3.5 text-muted-foreground whitespace-nowrap">
                          <span className="flex items-center gap-1.5 text-xs">
                            <Calendar className="h-3.5 w-3.5 flex-shrink-0" />
                            {formatDate(run.started_at)}
                          </span>
                        </td>

                        {/* Duration */}
                        <td className="px-4 py-3.5 text-muted-foreground text-xs tabular-nums">
                          {duration(run.started_at, run.completed_at)}
                        </td>

                        {/* Delta R² */}
                        <td className="px-4 py-3.5 min-w-[120px]">
                          {deltaR2 !== undefined ? (
                            <R2Indicator value={deltaR2} />
                          ) : (
                            <span className="text-xs text-muted-foreground">—</span>
                          )}
                        </td>

                        {/* Actions */}
                        <td className="px-4 py-3.5">
                          <div className="flex items-center gap-1">
                            {run.status === "completed" && (
                              <>
                                {!run.is_active && run.model_name && (
                                  <Button
                                    variant="ghost"
                                    size="sm"
                                    className="h-7 px-2 text-xs gap-1 text-primary"
                                    onClick={() => handleActivate(run)}
                                    disabled={activatingId === run.id}
                                    id={`activate-btn-${run.id}`}
                                  >
                                    {activatingId === run.id ? (
                                      <Loader2 className="h-3 w-3 animate-spin" />
                                    ) : (
                                      <Zap className="h-3 w-3" />
                                    )}
                                    Activate
                                  </Button>
                                )}
                                {run.is_active && (
                                  <span className="inline-flex items-center gap-1 text-xs font-medium text-amber-600 dark:text-amber-400 px-2">
                                    <Star className="h-3 w-3 fill-amber-500" />
                                    Active
                                  </span>
                                )}
                              </>
                            )}
                            {run.evaluation_report && (
                              <Button
                                variant="ghost"
                                size="sm"
                                className="h-7 px-2 text-xs gap-1 text-muted-foreground hover:text-foreground"
                                onClick={() => setExpandedRow(isExpanded ? null : run.id)}
                                id={`expand-run-btn-${run.id}`}
                              >
                                {isExpanded ? (
                                  <ChevronUp className="h-3 w-3" />
                                ) : (
                                  <ChevronDown className="h-3 w-3" />
                                )}
                                Report
                              </Button>
                            )}
                          </div>
                        </td>
                      </tr>

                      {/* Expanded Eval Summary */}
                      {isExpanded && run.evaluation_report && (
                        <tr key={`${run.id}-expanded`} className="border-b border-border/60 bg-muted/10">
                          <td colSpan={7} className="px-6 py-4">
                            <EvalSummaryRow report={run.evaluation_report} />
                          </td>
                        </tr>
                      )}
                    </>
                  )
                })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}
