"use client"

import { useState } from "react"
import Link from "next/link"
import {
  BrainCircuit,
  Play,
  BarChart3,
  ArrowLeft,
  Lock,
  Loader2,
  Layers2,
  Upload,
} from "lucide-react"
import { Header } from "@/components/header"
import { Button } from "@/components/ui/button"
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { TrainingTriggerPanel } from "@/components/training/training-trigger-panel"
import { EvaluationReportCard } from "@/components/training/evaluation-report-card"
import { ModelActivationDashboard } from "@/components/training/model-activation-dashboard"
import { ModelUploadPanel } from "@/components/training/model-upload-panel"
import { TrainingRunTable } from "@/components/training/training-run-table"
import { useAuth } from "@/components/auth-provider"
import { type TrainingRun } from "@/lib/types"

export default function ModelsPage() {
  const { user, loading } = useAuth()
  // Bump this to tell the run table to re-fetch after a new run starts
  /** Bumps only for Past runs table while Train Model tab is open (tab panels unmount when inactive). */
  const [tableRefreshTrigger, setTableRefreshTrigger] = useState(0)
  const [activeTab, setActiveTab] = useState("activation")

  function handleRunStarted(run: TrainingRun) {
    // If a run was submitted, is actively processing, or has finished, refresh the history table
    if (
      run.status === "queued" ||
      run.status === "running" ||
      run.status === "completed" ||
      run.status === "failed"
    ) {
      setTableRefreshTrigger((n) => n + 1)
    }
  }

  // ── Loading skeleton ──────────────────────────────────────────────────────
  if (loading) {
    return (
      <div className="min-h-screen bg-background">
        <Header />
        <div className="flex items-center justify-center py-40">
          <Loader2 className="h-8 w-8 animate-spin text-primary" />
        </div>
      </div>
    )
  }

  // ── Not logged in ─────────────────────────────────────────────────────────
  if (!user) {
    return (
      <div className="min-h-screen bg-background">
        <Header />
        <main className="container mx-auto max-w-4xl px-4 py-20 text-center">
          <div className="flex h-16 w-16 items-center justify-center rounded-2xl bg-muted mx-auto mb-4">
            <Lock className="h-8 w-8 text-muted-foreground" />
          </div>
          <h2 className="text-xl font-bold text-foreground">Authentication Required</h2>
          <p className="mt-2 text-sm text-muted-foreground">
            You must be logged in as an admin to access the training dashboard.
          </p>
          <Button asChild className="mt-6">
            <Link href="/login">Log in</Link>
          </Button>
        </main>
      </div>
    )
  }

  // ── Not an admin ──────────────────────────────────────────────────────────
  if (!user.is_admin) {
    return (
      <div className="min-h-screen bg-background">
        <Header />
        <main className="container mx-auto max-w-4xl px-4 py-20 text-center">
          <div className="flex h-16 w-16 items-center justify-center rounded-2xl bg-muted mx-auto mb-4">
            <Lock className="h-8 w-8 text-muted-foreground" />
          </div>
          <h2 className="text-xl font-bold text-foreground">Access Denied</h2>
          <p className="mt-2 text-sm text-muted-foreground">
            You must be logged in as an admin to access the training dashboard.
          </p>
          <Button asChild className="mt-6" variant="outline">
            <Link href="/">Return home</Link>
          </Button>
        </main>
      </div>
    )
  }
  // ── Main page ─────────────────────────────────────────────────────────────
  return (
    <div className="min-h-screen bg-background">
      <Header />

      <main className="container mx-auto max-w-7xl px-4 py-6 sm:py-8">
        {/* Page header */}
        <div className="mb-6 flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between">
          <div>
            <div className="flex items-center gap-2 mb-1">
              <Button
                variant="ghost"
                size="sm"
                className="gap-1.5 text-muted-foreground hover:text-foreground -ml-2 h-8"
                asChild
              >
                <Link href="/">
                  <ArrowLeft className="h-4 w-4" />
                  Back
                </Link>
              </Button>
            </div>
            <div className="flex items-center gap-3">
              <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-primary/10 shadow-sm">
                <BrainCircuit className="h-5 w-5 text-primary" />
              </div>
              <div>
                <h2 className="text-xl font-bold tracking-tight text-foreground sm:text-2xl">
                  ML Dashboard
                </h2>
                <p className="text-sm text-muted-foreground mt-0.5">
                  Set the active model, upload artifacts, train, review run history, and read evaluation reports.
                </p>
              </div>
            </div>
          </div>

          {/* Admin badge */}
          <div className="inline-flex items-center gap-1.5 self-start rounded-full border border-amber-200 bg-amber-50 dark:border-amber-700 dark:bg-amber-900/20 px-3 py-1 text-xs font-semibold text-amber-700 dark:text-amber-400">
            <Lock className="h-3 w-3" />
            Admin Only
          </div>
        </div>

        {/* Tabs */}
        <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-5">
          <TabsList className="min-h-10 h-auto flex-wrap gap-0.5 p-1" id="models-tabs">
            <TabsTrigger value="activation" className="gap-1.5 text-sm" id="tab-activation">
              <Layers2 className="h-3.5 w-3.5" />
              Active model
            </TabsTrigger>
            <TabsTrigger value="upload" className="gap-1.5 text-sm" id="tab-upload">
              <Upload className="h-3.5 w-3.5" />
              Upload Model
            </TabsTrigger>
            <TabsTrigger value="trigger" className="gap-1.5 text-sm" id="tab-trigger">
              <Play className="h-3.5 w-3.5" />
              Train Model
            </TabsTrigger>
            <TabsTrigger value="evaluation" className="gap-1.5 text-sm" id="tab-evaluation">
              <BarChart3 className="h-3.5 w-3.5" />
              Model Evaluation
            </TabsTrigger>
          </TabsList>

          {/* Only the active panel mounts — avoids duplicate fetches from hidden Radix tabs + refresh-key bumps. */}
          <div className="mt-0">
            {activeTab === "activation" && (
              <div role="tabpanel" id="panel-activation" aria-labelledby="tab-activation">
                <ModelActivationDashboard tableRefreshTrigger={tableRefreshTrigger} />
              </div>
            )}
            {activeTab === "upload" && (
              <div role="tabpanel" id="panel-upload" aria-labelledby="tab-upload">
                <ModelUploadPanel />
              </div>
            )}
            {activeTab === "trigger" && (
              <div role="tabpanel" id="panel-trigger" aria-labelledby="tab-trigger">
                <div className="space-y-10">
                  <div className="grid grid-cols-1 gap-6 lg:grid-cols-5">
                    <div className="lg:col-span-3">
                      <TrainingTriggerPanel onRunStarted={handleRunStarted} />
                    </div>

                    <div className="lg:col-span-2">
                      <div className="rounded-2xl border border-border bg-card shadow-sm p-5">
                        <h3 className="text-sm font-semibold text-foreground mb-3">Pipeline Steps</h3>
                        <div className="space-y-2.5 text-sm">
                          {[
                            { step: "1", title: "Download Data", desc: "Historical prices via yfinance" },
                            { step: "2", title: "Generate Samples", desc: "Grid search finds ground-truth δ/θ" },
                            { step: "3", title: "Train Random Forest", desc: "Scikit-learn RF logged to MLflow" },
                            { step: "4", title: "Evaluate", desc: "Direct metrics + end-to-end quality" },
                          ].map((s) => (
                            <div key={s.step} className="flex gap-3 items-start">
                              <div className="flex h-5 w-5 flex-shrink-0 items-center justify-center rounded-full bg-primary text-[10px] font-bold text-primary-foreground mt-0.5">
                                {s.step}
                              </div>
                              <div>
                                <p className="font-medium text-foreground">{s.title}</p>
                                <p className="text-xs text-muted-foreground">{s.desc}</p>
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                    </div>
                  </div>

                  <section className="space-y-3">
                    <h3 className="text-base font-semibold text-foreground">Past runs</h3>
                    <TrainingRunTable refreshTrigger={tableRefreshTrigger} />
                  </section>
                </div>
              </div>
            )}
            {activeTab === "evaluation" && (
              <div role="tabpanel" id="panel-evaluation" aria-labelledby="tab-evaluation">
                <EvaluationReportCard />
              </div>
            )}
          </div>
        </Tabs>
      </main>
    </div>
  )
}
