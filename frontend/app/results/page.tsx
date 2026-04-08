"use client"

import { useEffect, useState, Suspense } from "react"
import { useRouter, useSearchParams } from "next/navigation"
import { ArrowLeft, Loader2 } from "lucide-react"
import { Header } from "@/components/header"
import { VisualizationResults } from "@/components/graphs-results"
import { Button } from "@/components/ui/button"
import { InfoTooltip } from "@/components/info-tooltip"
import { type GeneratedData, type JobParameters } from "@/lib/types"
import { useAuth } from "@/components/auth-provider"

function FidelityCard({ score, csvSimilarity }: { score: number; csvSimilarity?: number }) {
  const r = 28
  const circ = 2 * Math.PI * r
  const safeScore = Math.max(0, Math.min(100, Math.round(score)))
  const offset = circ * (1 - safeScore / 100)
  const color = safeScore >= 75 ? "#22c55e" : safeScore >= 50 ? "#f59e0b" : "#ef4444"
  const label = safeScore >= 75 ? "Excellent match" : safeScore >= 50 ? "Moderate match" : "Low match"

  return (
    <div className="h-full rounded-xl border border-border bg-card px-5 py-4">
      <div className="flex h-full flex-col justify-center gap-1">
        <div className="flex items-center gap-1">
          <p className="text-xs font-semibold uppercase tracking-widest text-muted-foreground">Fidelity</p>
          <InfoTooltip
            content={[
              "What these scores mean",
              "Intent fidelity: Did we match your intent?",
              "CSV similarity: How close is the result to your input overall?",
            ]}
          />
        </div>
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
                {safeScore}
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
            {typeof csvSimilarity === "number" && Number.isFinite(csvSimilarity) && (
              <p className="text-[10px] text-muted-foreground">CSV sim: {Math.round(csvSimilarity)}%</p>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

function ResultsContent() {
  const router = useRouter()
  const searchParams = useSearchParams()
  const { user } = useAuth()

  const jobId = searchParams.get("jobId")
  const fromReports = !!jobId
  const backHref = fromReports ? "/reports" : "/"

  const [data, setData] = useState<GeneratedData | null>(null)
  const [jobParams, setJobParams] = useState<JobParameters | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (jobId) {
      // Wait until auth has resolved — user.id is needed for the status URL
      if (!user?.id) return
      fetch(`/api/status/user/${user.id}/${jobId}`)
        .then((r) => r.json())
        .then((d) => {
          if (d.chart_data) {
            setData(d.chart_data as GeneratedData)
            const raw = localStorage.getItem(`goldenduck_job_params_${jobId}`)
            if (raw) setJobParams(JSON.parse(raw) as JobParameters)
          } else {
            setError("No visualisation data available for this job.")
          }
        })
        .catch(() => setError("Failed to load report."))
        .finally(() => setLoading(false))
      return
    }

    // Legacy path: the parameterization form stored chart_data in localStorage
    const raw = localStorage.getItem("goldenduck_chart_data")
    if (!raw) {
      router.replace("/")
      return
    }
    try {
      setData(JSON.parse(raw) as GeneratedData)
    } catch {
      router.replace("/")
      return
    }
    setLoading(false)
  }, [jobId, user?.id, router])

  return (
    <div className="min-h-screen bg-background">
      <Header />
      <main className="container mx-auto max-w-7xl px-4 py-6 sm:py-8">
        <div className="mb-6 flex items-center justify-between">
          <div>
            <h2 className="text-xl font-bold tracking-tight text-foreground sm:text-2xl">
              Results
            </h2>
            <p className="mt-1 text-sm text-muted-foreground sm:text-base">
              Synthetic data generation complete.
            </p>
          </div>
          <Button
            variant="outline"
            size="sm"
            className="gap-2"
            onClick={() => router.push(backHref)}
          >
            <ArrowLeft className="h-4 w-4" />
            {fromReports ? "Back to Reports" : "Back"}
          </Button>
        </div>

        {loading && (
          <div className="flex h-64 items-center justify-center">
            <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
          </div>
        )}

        {!loading && error && (
          <div className="flex h-64 flex-col items-center justify-center gap-3 text-center">
            <p className="text-muted-foreground">{error}</p>
            <Button variant="outline" size="sm" onClick={() => router.push(backHref)}>
              <ArrowLeft className="h-4 w-4 mr-2" />
              {fromReports ? "Back to Reports" : "Back"}
            </Button>
          </div>
        )}

        {!loading && !error && data && (
          <>
            {jobParams && (
              <div className="mb-6 grid gap-4 lg:grid-cols-[1fr_240px] lg:items-stretch">
                <div className="rounded-xl border border-border bg-card px-5 py-4">
                  <h3 className="mb-3 text-sm font-semibold uppercase tracking-wider text-foreground">Input Parameters</h3>
                  <div className="grid grid-cols-2 gap-4 sm:grid-cols-3 lg:grid-cols-6">
                    {[
                      { label: "Volatility", value: jobParams.volatility.toFixed(2), range: "Range: 0.5 – 2.0" },
                      { label: "Trend", value: jobParams.trend.toFixed(2), range: "Range: −1.0 – 1.0" },
                      { label: "Fat Tails", value: jobParams.fatTails.toFixed(2), range: "Range: 0.5 – 2.0" },
                      { label: "Momentum", value: jobParams.momentum.toFixed(2), range: "Range: 0.0 – 1.0" },
                      { label: "Time Horizon", value: `${jobParams.timeHorizon}d`, range: "Range: 500 – 2600 days" },
                      { label: "File", value: jobParams.fileName ?? "—", range: "" },
                    ].map((item) => (
                      <div key={item.label}>
                        <div className="mb-0.5 flex items-center gap-1">
                          <p className="text-[10px] font-semibold uppercase tracking-widest text-muted-foreground">{item.label}</p>
                          {item.range && <InfoTooltip content={item.range} />}
                        </div>
                        <p className="truncate font-mono text-sm font-medium text-foreground" title={item.value}>
                          {item.value}
                        </p>
                      </div>
                    ))}
                  </div>
                </div>

                <FidelityCard
                  score={
                    typeof data.intentFidelity === "number" && Number.isFinite(data.intentFidelity)
                      ? data.intentFidelity * 100
                      : typeof data.selectionScore === "number" && Number.isFinite(data.selectionScore)
                        ? Math.max(0, Math.min(100, 100 * (1 - data.selectionScore)))
                        : typeof data.overallMatch === "number" && Number.isFinite(data.overallMatch)
                          ? data.overallMatch * 100
                          : 0
                  }
                  csvSimilarity={
                    typeof data.csvSimilarity === "number" && Number.isFinite(data.csvSimilarity)
                      ? data.csvSimilarity * 100
                      : undefined
                  }
                />
              </div>
            )}
            <VisualizationResults data={data} />
          </>
        )}
      </main>
    </div>
  )
}

// useSearchParams() requires a Suspense boundary in the App Router
export default function ResultsPage() {
  return (
    <Suspense fallback={null}>
      <ResultsContent />
    </Suspense>
  )
}
