"use client"

import { useCallback, useEffect, useState } from "react"
import { Loader2, RefreshCw, Star, Zap } from "lucide-react"
import { Button } from "@/components/ui/button"
import { useToast } from "@/components/ui/use-toast"
import type { TrainingRun, UploadedModelRow } from "@/lib/types"
import { fetchTrainingGet } from "@/lib/training-fetch"

type ModelSource = "upload" | "trained"

export interface UnifiedActivatableModel {
  modelName: string
  sources: ModelSource[]
  createdAt: string
  isActive: boolean
}

function mergeActivatableModels(
  uploads: UploadedModelRow[],
  runs: TrainingRun[],
): UnifiedActivatableModel[] {
  const map = new Map<string, UnifiedActivatableModel>()

  const newer = (a: string, b: string) =>
    new Date(a) >= new Date(b) ? a : b

  for (const u of uploads) {
    map.set(u.model_name, {
      modelName: u.model_name,
      sources: ["upload"],
      createdAt: u.created_at,
      isActive: u.is_active,
    })
  }

  for (const r of runs) {
    if (r.status !== "completed" || !r.model_name) continue
    const ts = r.completed_at ?? r.started_at
    const prev = map.get(r.model_name)
    if (!prev) {
      map.set(r.model_name, {
        modelName: r.model_name,
        sources: ["trained"],
        createdAt: ts,
        isActive: !!r.is_active,
      })
    } else {
      if (!prev.sources.includes("trained")) {
        prev.sources = [...prev.sources, "trained"]
      }
      prev.createdAt = newer(prev.createdAt, ts)
      prev.isActive = prev.isActive || !!r.is_active
    }
  }

  return [...map.values()].sort(
    (a, b) => new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime(),
  )
}

function formatDate(iso: string) {
  return new Intl.DateTimeFormat("en-SG", {
    dateStyle: "medium",
    timeStyle: "short",
  }).format(new Date(iso))
}

function SourceBadges({ sources }: { sources: ModelSource[] }) {
  const labels: Record<ModelSource, string> = {
    upload: "Upload",
    trained: "Trained",
  }
  return (
    <div className="flex flex-wrap gap-1.5">
      {sources.map((s) => (
        <span
          key={s}
          className="inline-flex rounded-md border border-border bg-muted/50 px-2 py-0.5 text-[11px] font-medium text-muted-foreground"
        >
          {labels[s]}
        </span>
      ))}
    </div>
  )
}

interface Props {
  refreshTrigger: number
}

export function ActiveModelTable({ refreshTrigger }: Props) {
  const { toast } = useToast()
  const [rows, setRows] = useState<UnifiedActivatableModel[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [activatingName, setActivatingName] = useState<string | null>(null)

  const fetchAll = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const [upRes, runRes] = await Promise.all([
        fetchTrainingGet("/api/training/uploads"),
        fetchTrainingGet("/api/training/runs"),
      ])
      if (!upRes.ok) throw new Error(`Uploads: ${upRes.status}`)
      if (!runRes.ok) throw new Error(`Runs: ${runRes.status}`)
      const upData = await upRes.json()
      const runData = await runRes.json()
      const uploads: UploadedModelRow[] = upData.uploads ?? []
      const runs: TrainingRun[] = runData.runs ?? []
      setRows(mergeActivatableModels(uploads, runs))
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to load models")
      setRows([])
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    fetchAll()
  }, [fetchAll, refreshTrigger])

  async function handleActivate(modelName: string) {
    if (activatingName) return
    setActivatingName(modelName)
    try {
      const res = await fetch(
        `/api/training/models/${encodeURIComponent(modelName)}/activate`,
        { method: "POST" },
      )
      if (!res.ok) throw new Error("Failed to activate")
      toast({ title: "Active model updated", description: modelName })
      await fetchAll()
    } catch {
      toast({
        title: "Error",
        description: "Could not activate this model.",
        variant: "destructive",
      })
    } finally {
      setActivatingName(null)
    }
  }

  return (
    <div className="space-y-4">
      <div className="flex justify-end">
        <Button
          variant="outline"
          size="sm"
          className="gap-1.5 h-8"
          onClick={fetchAll}
          disabled={loading}
          type="button"
        >
          <RefreshCw className={`h-3.5 w-3.5 ${loading ? "animate-spin" : ""}`} />
          Refresh
        </Button>
      </div>

      <div className="rounded-2xl border border-border bg-card shadow-sm overflow-hidden">
        {error && (
          <div className="border-b border-border bg-destructive/10 px-4 py-3 text-sm text-destructive">
            {error}
          </div>
        )}
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-border bg-muted/40">
                <th className="px-4 py-3 text-left text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                  Model
                </th>
                <th className="px-4 py-3 text-left text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                  Source
                </th>
                <th className="px-4 py-3 text-left text-xs font-semibold uppercase tracking-wider text-muted-foreground whitespace-nowrap">
                  Added
                </th>
                <th className="px-4 py-3 text-right text-xs font-semibold uppercase tracking-wider text-muted-foreground w-[140px]">
                  Action
                </th>
              </tr>
            </thead>
            <tbody>
              {loading && (
                <tr>
                  <td colSpan={4} className="px-4 py-16 text-center">
                    <Loader2 className="h-8 w-8 animate-spin text-primary mx-auto" />
                  </td>
                </tr>
              )}
              {!loading && rows.length === 0 && (
                <tr>
                  <td colSpan={4} className="px-4 py-14 text-center text-sm text-muted-foreground">
                    No models yet. Upload a <span className="font-mono text-xs">.pkl</span> or complete a
                    training run.
                  </td>
                </tr>
              )}
              {!loading &&
                rows.map((row) => (
                  <tr key={row.modelName} className="border-b border-border/60 last:border-0">
                    <td className="px-4 py-3.5 font-mono text-xs text-foreground">{row.modelName}</td>
                    <td className="px-4 py-3.5">
                      <SourceBadges sources={row.sources} />
                    </td>
                    <td className="px-4 py-3.5 text-muted-foreground whitespace-nowrap text-xs">
                      {formatDate(row.createdAt)}
                    </td>
                    <td className="px-4 py-3.5 text-right">
                      {row.isActive ? (
                        <span className="inline-flex items-center justify-end gap-1 text-xs font-medium text-amber-600 dark:text-amber-400">
                          <Star className="h-3.5 w-3.5 fill-amber-500" />
                          Active
                        </span>
                      ) : (
                        <Button
                          type="button"
                          variant="ghost"
                          size="sm"
                          className="h-8 gap-1 text-xs text-primary"
                          onClick={() => handleActivate(row.modelName)}
                          disabled={activatingName === row.modelName}
                        >
                          {activatingName === row.modelName ? (
                            <Loader2 className="h-3.5 w-3.5 animate-spin" />
                          ) : (
                            <Zap className="h-3.5 w-3.5" />
                          )}
                          Activate
                        </Button>
                      )}
                    </td>
                  </tr>
                ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}
