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

function CountUpValue({ target, decimals = 2, suffix = "", duration = 1600 }: {
  target: number
  decimals?: number
  suffix?: string
  duration?: number
}) {
  const [display, setDisplay] = useState(0)

  useEffect(() => {
    const start = performance.now()
    const tick = (now: number) => {
      const progress = Math.min((now - start) / duration, 1)
      const eased = 1 - Math.pow(1 - progress, 3)
      setDisplay(target * eased)
      if (progress < 1) requestAnimationFrame(tick)
    }
    requestAnimationFrame(tick)
  }, [target, duration])

  return <>{display.toFixed(decimals)}{suffix}</>
}

function FidelityCard({ score, csvSimilarity }: { score: number; csvSimilarity?: number }) {
  const r = 22
  const circ = 2 * Math.PI * r
  const safeScore = Math.max(0, Math.min(100, Math.round(score)))
  const offset = circ * (1 - safeScore / 100)
  const color = safeScore >= 75 ? "#22c55e" : safeScore >= 50 ? "#f59e0b" : "#ef4444"
  const label = safeScore >= 75 ? "Excellent match" : safeScore >= 50 ? "Moderate match" : "Low match"

  return (
    <div className="flex flex-col justify-between pl-8 border-l border-border">
      <p className="text-[10px] font-semibold uppercase tracking-widest text-muted-foreground">Fidelity</p>
      <div className="flex items-center gap-3 mt-2">
        <div className="relative h-12 w-12 shrink-0">
          <svg viewBox="0 0 64 64" className="h-12 w-12 -rotate-90">
            <circle cx="32" cy="32" r={r} fill="none" stroke="var(--border)" strokeWidth="5" />
            <circle
              cx="32"
              cy="32"
              r={r}
              fill="none"
              stroke={color}
              strokeWidth="5"
              strokeDasharray={circ}
              strokeDashoffset={offset}
              strokeLinecap="round"
              style={{ transition: "stroke-dashoffset 1.2s ease-out" }}
            />
          </svg>
        </div>
        <div>
          <p className="text-2xl font-light text-foreground tabular-nums leading-none">{safeScore}</p>
          <p className="text-[11px] font-medium mt-1" style={{ color }}>{label}</p>
          {typeof csvSimilarity === "number" && Number.isFinite(csvSimilarity) && (
            <p className="text-[10px] text-muted-foreground mt-0.5">CSV sim: {Math.round(csvSimilarity)}%</p>
          )}
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
              <div className="mb-6 px-1 py-5 flex items-center gap-10">
                {/* Input Parameters */}
                <div className="shrink-0">
                  <p className="text-[10px] font-semibold uppercase tracking-widest text-muted-foreground mb-3">
                    Input Parameters
                  </p>
                  <div className="flex divide-x divide-border">
                    {([
                      { label: "Volatility", target: jobParams.volatility, decimals: 2, suffix: "", range: "Range: 0.5 – 2.0" },
                      { label: "Trend", target: jobParams.trend, decimals: 2, suffix: "", range: "Range: −1.0 – 1.0" },
                      { label: "Fat Tails", target: jobParams.fatTails, decimals: 2, suffix: "", range: "Range: 0.5 – 2.0" },
                      { label: "Momentum", target: jobParams.momentum, decimals: 2, suffix: "", range: "Range: 0.0 – 1.0" },
                      { label: "Time Horizon", target: jobParams.timeHorizon, decimals: 0, suffix: "d", range: "Range: 500 – 2600 days" },
                    ] as const).map((item) => (
                      <div key={item.label} className="px-5 first:pl-0">
                        <div className="mb-1 flex items-center gap-1">
                          <p className="text-[10px] font-medium uppercase tracking-widest text-muted-foreground whitespace-nowrap">
                            {item.label}
                          </p>
                          <InfoTooltip content={item.range} />
                        </div>
                        <p className="text-xl font-light text-foreground whitespace-nowrap tabular-nums">
                          <CountUpValue target={item.target} decimals={item.decimals} suffix={item.suffix} />
                        </p>
                      </div>
                    ))}
                    <div className="px-5">
                      <p className="mb-1 text-[10px] font-medium uppercase tracking-widest text-muted-foreground whitespace-nowrap">File</p>
                      <p className="text-xl font-light text-foreground whitespace-nowrap">{jobParams.fileName ?? "—"}</p>
                    </div>
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
