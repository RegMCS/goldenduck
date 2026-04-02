"use client"

import { useState } from "react"
import {
  Play,
  Loader2,
  CheckCircle2,
  XCircle,
  ChevronRight,
  Database,
  Cpu,
  BarChart3,
  Clock,
} from "lucide-react"
import { Button } from "@/components/ui/button"
import { Switch } from "@/components/ui/switch"
import { Label } from "@/components/ui/label"
import { Slider } from "@/components/ui/slider"
import { type TrainingRun } from "@/lib/types"

// Coarse step labels matching training_worker.py publications
const STEPS = [
  { index: 1, label: "Downloading historical data", icon: Database },
  { index: 2, label: "Generating training samples", icon: Cpu },
  { index: 3, label: "Training Random Forest models", icon: Cpu },
  { index: 4, label: "Running model evaluation", icon: BarChart3 },
]

function estimateDuration(nAssets: number, nScenarios: number, testingMode: boolean): string {
  if (testingMode) return "~30–60 min"
  const sampleCount = nAssets * nScenarios
  if (sampleCount <= 2_000) return "~1–2 hours"
  if (sampleCount <= 10_000) return "~4–6 hours"
  return "~12+ hours"
}

interface Props {
  onRunStarted: (run: TrainingRun) => void
}

export function TrainingTriggerPanel({ onRunStarted }: Props) {
  const [testingMode, setTestingMode] = useState(true)
  const [nAssets, setNAssets] = useState(50)
  const [nScenarios, setNScenarios] = useState(10)

  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [activeRun, setActiveRun] = useState<TrainingRun | null>(null)

  const estimated = estimateDuration(nAssets, nScenarios, testingMode)

  async function handleStart() {
    setSubmitting(true)
    setError(null)

    try {
      const res = await fetch("/api/training/start", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          testing_mode: testingMode,
          n_assets: nAssets,
          n_scenarios: nScenarios,
          run_evaluation: true,
        }),
      })

      const data = await res.json()
      if (!res.ok) throw new Error(data.detail ?? "Failed to start training")

      // data is the newly created TrainingRun
      setActiveRun(data)
      onRunStarted(data)

      // Poll progress
      pollProgress(data.id)
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unknown error")
    } finally {
      setSubmitting(false)
    }
  }

  async function pollProgress(runId: string) {
    const interval = setInterval(async () => {
      try {
        const res = await fetch(`/api/training/runs/${runId}`)
        if (!res.ok) return
        const run: TrainingRun = await res.json()
        setActiveRun(run)
        if (run.status === "completed" || run.status === "failed") {
          clearInterval(interval)
          onRunStarted(run)
        }
      } catch {
        // ignore transient errors
      }
    }, 5_000)
  }

  const currentStepIndex = activeRun?.step_index ?? 0

  return (
    <div className="space-y-6">
      {/* Config panel */}
      {!activeRun && (
        <div className="rounded-2xl border border-border bg-card shadow-sm p-6 space-y-6">
          <div>
            <h3 className="text-base font-semibold text-foreground">Training Configuration</h3>
            <p className="text-sm text-muted-foreground mt-0.5">
              Configure the dataset size and mode. Evaluation will run automatically after training completes.
            </p>
          </div>

          {/* Testing mode toggle */}
          <div className="flex items-center justify-between rounded-xl border border-border bg-muted/30 px-4 py-3">
            <div>
              <Label htmlFor="testing-mode-switch" className="text-sm font-medium">Testing Mode</Label>
              <p className="text-xs text-muted-foreground mt-0.5">
                Uses reduced assets &amp; scenarios for a fast ~30-min run
              </p>
            </div>
            <Switch
              id="testing-mode-switch"
              checked={testingMode}
              onCheckedChange={setTestingMode}
            />
          </div>

          {/* N Assets */}
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <Label className="text-sm font-medium">Assets to Download</Label>
              <span className="text-sm font-bold text-primary tabular-nums">{nAssets}</span>
            </div>
            <Slider
              id="n-assets-slider"
              min={testingMode ? 10 : 100}
              max={testingMode ? 100 : 500}
              step={testingMode ? 10 : 50}
              value={[nAssets]}
              onValueChange={([v]) => setNAssets(v)}
              className="w-full"
            />
            <div className="flex justify-between text-xs text-muted-foreground">
              <span>{testingMode ? "10 (fastest)" : "100"}</span>
              <span>{testingMode ? "100" : "500 (most accurate)"}</span>
            </div>
          </div>

          {/* N Scenarios */}
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <Label className="text-sm font-medium">Scenarios per Asset</Label>
              <span className="text-sm font-bold text-primary tabular-nums">{nScenarios}</span>
            </div>
            <Slider
              id="n-scenarios-slider"
              min={testingMode ? 5 : 10}
              max={testingMode ? 20 : 50}
              step={5}
              value={[nScenarios]}
              onValueChange={([v]) => setNScenarios(v)}
              className="w-full"
            />
            <div className="flex justify-between text-xs text-muted-foreground">
              <span>{testingMode ? "5" : "10"}</span>
              <span>{testingMode ? "20" : "50"}</span>
            </div>
          </div>

          {/* Summary strip */}
          <div className="grid grid-cols-3 gap-3">
            {[
              { label: "Total Samples", value: `${(nAssets * nScenarios).toLocaleString()}` },
              { label: "Grid Combos", value: "24 / sample" },
              { label: "Est. Duration", value: estimated },
            ].map((item) => (
              <div key={item.label} className="rounded-xl border border-border bg-background px-3 py-2.5 text-center">
                <p className="text-xs text-muted-foreground">{item.label}</p>
                <p className="text-sm font-semibold text-foreground mt-0.5">{item.value}</p>
              </div>
            ))}
          </div>

          {error && (
            <div className="flex items-center gap-2 rounded-lg border border-destructive/30 bg-destructive/10 px-4 py-3 text-sm text-destructive">
              <XCircle className="h-4 w-4 flex-shrink-0" />
              {error}
            </div>
          )}

          <Button
            id="start-training-btn"
            className="w-full gap-2"
            size="lg"
            onClick={handleStart}
            disabled={submitting}
          >
            {submitting ? (
              <Loader2 className="h-4 w-4 animate-spin" />
            ) : (
              <Play className="h-4 w-4" />
            )}
            {submitting ? "Submitting…" : "Start Training"}
          </Button>
        </div>
      )}

      {/* Live progress card */}
      {activeRun && (
        <div className="rounded-2xl border border-border bg-card shadow-sm p-6 space-y-5">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-base font-semibold text-foreground">Training in Progress</h3>
              <p className="text-xs text-muted-foreground font-mono mt-0.5">
                Run ID: {activeRun.id.slice(0, 12)}…
              </p>
            </div>
            {activeRun.status === "running" && (
              <span className="inline-flex items-center gap-1.5 rounded-full border border-blue-200 bg-blue-100 px-2.5 py-1 text-xs font-medium text-blue-700 dark:border-blue-700 dark:bg-blue-900/30 dark:text-blue-400">
                <Loader2 className="h-3 w-3 animate-spin" />
                Running
              </span>
            )}
            {activeRun.status === "completed" && (
              <span className="inline-flex items-center gap-1.5 rounded-full border border-emerald-200 bg-emerald-100 px-2.5 py-1 text-xs font-medium text-emerald-700 dark:border-emerald-700 dark:bg-emerald-900/30 dark:text-emerald-400">
                <CheckCircle2 className="h-3 w-3" />
                Completed
              </span>
            )}
            {activeRun.status === "failed" && (
              <span className="inline-flex items-center gap-1.5 rounded-full border border-red-200 bg-red-100 px-2.5 py-1 text-xs font-medium text-red-700 dark:border-red-700 dark:bg-red-900/30 dark:text-red-400">
                <XCircle className="h-3 w-3" />
                Failed
              </span>
            )}
          </div>

          {/* Step list */}
          <div className="space-y-2">
            {STEPS.map((step) => {
              const isDone = activeRun.status === "completed" || currentStepIndex > step.index
              const isCurrent = activeRun.status === "running" && currentStepIndex === step.index
              const StepIcon = step.icon

              return (
                <div
                  key={step.index}
                  className={`flex items-center gap-3 rounded-xl px-4 py-3 transition-colors ${
                    isDone
                      ? "bg-emerald-50 dark:bg-emerald-900/20"
                      : isCurrent
                      ? "bg-blue-50 dark:bg-blue-900/20"
                      : "bg-muted/30"
                  }`}
                >
                  <div
                    className={`flex h-7 w-7 flex-shrink-0 items-center justify-center rounded-full ${
                      isDone
                        ? "bg-emerald-500"
                        : isCurrent
                        ? "bg-blue-500"
                        : "bg-muted"
                    }`}
                  >
                    {isDone ? (
                      <CheckCircle2 className="h-4 w-4 text-white" />
                    ) : isCurrent ? (
                      <Loader2 className="h-4 w-4 text-white animate-spin" />
                    ) : (
                      <StepIcon className="h-4 w-4 text-muted-foreground" />
                    )}
                  </div>
                  <div className="flex-1 min-w-0">
                    <p
                      className={`text-sm font-medium truncate ${
                        isDone
                          ? "text-emerald-700 dark:text-emerald-400"
                          : isCurrent
                          ? "text-blue-700 dark:text-blue-400"
                          : "text-muted-foreground"
                      }`}
                    >
                      Step {step.index} of {STEPS.length}: {step.label}
                    </p>
                  </div>
                  {isCurrent && (
                    <Clock className="h-3.5 w-3.5 text-blue-400 flex-shrink-0 animate-pulse" />
                  )}
                  {isDone && (
                    <ChevronRight className="h-3.5 w-3.5 text-emerald-400 flex-shrink-0" />
                  )}
                </div>
              )
            })}
          </div>

          {activeRun.status === "failed" && activeRun.error && (
            <div className="rounded-lg border border-destructive/30 bg-destructive/10 px-4 py-3 text-sm text-destructive">
              <p className="font-medium">Training failed</p>
              <p className="mt-1 font-mono text-xs break-all">{activeRun.error}</p>
            </div>
          )}

          {activeRun.status === "completed" && activeRun.model_name && (
            <div className="rounded-xl border border-emerald-200 bg-emerald-50 dark:border-emerald-700 dark:bg-emerald-900/20 px-4 py-3 text-sm text-emerald-700 dark:text-emerald-400">
              <p className="font-medium">✓ Model saved</p>
              <p className="mt-0.5 font-mono text-xs">{activeRun.model_name}</p>
            </div>
          )}

          {(activeRun.status === "completed" || activeRun.status === "failed") && (
            <Button
              variant="outline"
              className="w-full"
              onClick={() => setActiveRun(null)}
            >
              Start Another Run
            </Button>
          )}
        </div>
      )}
    </div>
  )
}
