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

interface JobMetrics {
  mean_match?: number
  volatility_match?: number
  kurtosis_match?: number
  acf_match?: number
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
  const [jobMetrics, setJobMetrics] = useState<JobMetrics | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const fmtMatch = (v?: number) => {
    if (typeof v !== "number" || !Number.isFinite(v)) return null
    const pct = Math.max(0, Math.min(1, v)) * 100
    return `${pct.toFixed(1)}%`
  }

  useEffect(() => {
    if (jobId) {
      // Wait until auth has resolved — user.id is needed for the status URL
      if (!user?.id) return
      fetch(`/api/status/user/${user.id}/${jobId}`)
        .then((r) => r.json())
        .then((d) => {
          if (d.chart_data) {
            setData(d.chart_data as GeneratedData)
            setJobMetrics((d.metrics ?? null) as JobMetrics | null)
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
              <div className="mb-6 rounded-xl border border-border bg-card px-5 py-4">
                <h3 className="text-sm font-semibold text-foreground mb-3">Input Parameters</h3>
                <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-4">
                  {[
                    { label: "Volatility", value: jobParams.volatility.toFixed(2), range: "Range: 0.5 – 2.0", match: fmtMatch(jobMetrics?.volatility_match) },
                    { label: "Trend", value: jobParams.trend.toFixed(2), range: "Range: −1.0 – 1.0", match: fmtMatch(jobMetrics?.mean_match) },
                    { label: "Fat Tails", value: jobParams.fatTails.toFixed(2), range: "Range: 0.5 – 2.0", match: fmtMatch(jobMetrics?.kurtosis_match) },
                    { label: "V Momentum", value: jobParams.momentum.toFixed(2), range: "Range: 0.0 – 1.0", match: fmtMatch(jobMetrics?.acf_match) },
                    { label: "Time Horizon", value: `${jobParams.timeHorizon} days`, range: "Range: 500 – 2600 days" },
                    ...(jobParams.fileName ? [{ label: "Input File", value: jobParams.fileName, range: "" }] : []),
                  ].map((item) => (
                    <div key={item.label}>
                      <div className="flex items-center gap-1 mb-0.5">
                        <p className="text-[10px] font-semibold text-muted-foreground uppercase tracking-widest">{item.label}</p>
                        {item.match && (
                          <span className="text-[10px] font-semibold text-emerald-600 dark:text-emerald-400">
                            ({item.match})
                          </span>
                        )}
                        {item.range && <InfoTooltip content={item.range} />}
                      </div>
                      <p className="text-sm font-mono font-medium text-foreground truncate" title={item.value}>{item.value}</p>
                    </div>
                  ))}
                </div>
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
