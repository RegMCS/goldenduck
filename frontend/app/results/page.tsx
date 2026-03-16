"use client"

import { useEffect, useState, Suspense } from "react"
import { useRouter, useSearchParams } from "next/navigation"
import { ArrowLeft, Loader2 } from "lucide-react"
import { Header } from "@/components/header"
import { VisualizationResults } from "@/components/graphs-results"
import { Button } from "@/components/ui/button"
import { type GeneratedData } from "@/lib/types"
import { useAuth } from "@/components/auth-provider"

function ResultsContent() {
  const router = useRouter()
  const searchParams = useSearchParams()
  const { user } = useAuth()

  const jobId = searchParams.get("jobId")
  const fromReports = !!jobId
  const backHref = fromReports ? "/reports" : "/"

  const [data, setData] = useState<GeneratedData | null>(null)
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

        {!loading && !error && data && <VisualizationResults data={data} />}
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
