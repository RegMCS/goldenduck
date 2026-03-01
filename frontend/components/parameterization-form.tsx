"use client"

import React, { useState, useRef } from "react"
import { useRouter } from "next/navigation"
import { Upload, X, FileText, Sparkles, RotateCcw, Download, AlertCircle, CheckCircle2, Loader2 } from "lucide-react"
import { Input } from "@/components/ui/input"
import { Button } from "@/components/ui/button"
import { Label } from "@/components/ui/label"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import {
  type MarketParameters,
  type GeneratedData,
  defaultParameters,
  REQUIRED_CSV_HEADERS,
} from "@/lib/types"

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? ""
const DEFAULT_TICKER = "AAPL"

type JobStatus = "queued" | "running" | "completed" | "failed"

const UUID_RE = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i

function getUserId(): string {
  if (typeof window === "undefined") return "anonymous"
  let id = localStorage.getItem("goldenduck_user_id")
  if (!id || !UUID_RE.test(id)) {
    id = crypto.randomUUID()
    localStorage.setItem("goldenduck_user_id", id)
  }
  return id
}

interface ParameterFieldProps {
  label: string
  description: string
  value: number
  onChange: (value: number) => void
  min: number
  max: number
  step: number
}

function ParameterField({ label, description, value, onChange, min, max, step }: ParameterFieldProps) {
  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <Label className="text-sm font-medium text-foreground">{label}</Label>
        <span className="text-xs text-muted-foreground">{min} – {max}</span>
      </div>
      <Input
        type="number"
        value={value}
        onChange={(e) => {
          const val = parseFloat(e.target.value)
          if (!isNaN(val)) {
            onChange(Math.min(max, Math.max(min, val)))
          }
        }}
        min={min}
        max={max}
        step={step}
      />
      <p className="text-xs text-muted-foreground">{description}</p>
    </div>
  )
}

export function ParameterizationForm({ onDataReady }: { onDataReady?: (data: GeneratedData) => void }) {
  const router = useRouter()
  const [parameters, setParameters] = useState<MarketParameters>(defaultParameters)
  const [isGenerating, setIsGenerating] = useState(false)
  const [fileError, setFileError] = useState<string | null>(null)
  const [jobStatus, setJobStatus] = useState<JobStatus | null>(null)
  const [downloadUrl, setDownloadUrl] = useState<string | null>(null)
  const [generateError, setGenerateError] = useState<string | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const pollIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null)

  const updateParameter = <K extends keyof MarketParameters>(
    key: K,
    value: MarketParameters[K]
  ) => {
    setParameters((prev) => ({ ...prev, [key]: value }))
  }

  const validateCsvHeaders = (file: File): Promise<boolean> => {
    return new Promise((resolve) => {
      const reader = new FileReader()
      reader.onload = (e) => {
        const text = e.target?.result as string
        const firstLine = text.split("\n")[0].toLowerCase().trim()
        const headers = firstLine.split(",").map((h) => h.trim())
        const hasAllHeaders = REQUIRED_CSV_HEADERS.every((required) =>
          headers.includes(required)
        )
        resolve(hasAllHeaders)
      }
      reader.onerror = () => resolve(false)
      reader.readAsText(file)
    })
  }

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return

    if (!file.name.endsWith(".csv")) {
      setFileError("Please upload a .csv file")
      updateParameter("inputFile", null)
      return
    }

    const valid = await validateCsvHeaders(file)
    if (!valid) {
      setFileError("CSV must have OHLCV headers: Open, High, Low, Close, Volume")
      updateParameter("inputFile", null)
      return
    }

    setFileError(null)
    updateParameter("inputFile", file)
  }

  const handleRemoveFile = () => {
    updateParameter("inputFile", null)
    setFileError(null)
    if (fileInputRef.current) {
      fileInputRef.current.value = ""
    }
  }

  const handleGenerate = async () => {
    if (!parameters.inputFile) {
      setGenerateError("Please upload a CSV file before generating.")
      return
    }
    setGenerateError(null)

    setIsGenerating(true)
    setJobStatus(null)
    setDownloadUrl(null)
    setGenerateError(null)

    const userId = getUserId()

    try {
      // Read CSV file content
      const csvContent = await new Promise<string>((resolve, reject) => {
        const reader = new FileReader()
        reader.onload = (e) => resolve(e.target?.result as string)
        reader.onerror = (e) => reject(e)
        reader.readAsText(parameters.inputFile!)
      })

      const res = await fetch(`${API_BASE}/api/generate/user/${userId}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          csv_data: csvContent,
          horizon: parameters.timeHorizon,
          desired_volatility: parameters.volatility,
          desired_trend: parameters.trend,
          desired_fat_tails: parameters.fatTails,
          desired_momentum: parameters.momentum,
        }),
      })

      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: "Failed to submit job" }))
        throw new Error(err.detail || "Failed to submit job")
      }

      const data = await res.json()
      setJobStatus("queued")

      pollIntervalRef.current = setInterval(async () => {
        try {
          const statusRes = await fetch(
            `${API_BASE}/api/status/user/${userId}/${data.job_id}`
          )
          const statusData = await statusRes.json()
          setJobStatus(statusData.status as JobStatus)

          if (statusData.status === "completed") {
            clearInterval(pollIntervalRef.current!)
            setDownloadUrl(`${API_BASE}/api/download/user/${userId}/${data.job_id}`)
            setIsGenerating(false)
            if (statusData.chart_data && onDataReady) {
              onDataReady(statusData.chart_data as GeneratedData)
            }
          } else if (statusData.status === "failed") {
            clearInterval(pollIntervalRef.current!)
            setGenerateError(statusData.error || "Job failed")
            setIsGenerating(false)
          }
        } catch {
          clearInterval(pollIntervalRef.current!)
          setGenerateError("Failed to poll job status")
          setIsGenerating(false)
        }
      }, 3000)
    } catch (err) {
      setGenerateError(err instanceof Error ? err.message : "Failed to submit job")
      setIsGenerating(false)
    }
  }

  const handleReset = () => {
    setParameters(defaultParameters)
    setFileError(null)
    setJobStatus(null)
    setDownloadUrl(null)
    setGenerateError(null)
    if (pollIntervalRef.current) clearInterval(pollIntervalRef.current)
    if (fileInputRef.current) {
      fileInputRef.current.value = ""
    }
  }

  return (
    <div className="grid gap-6 lg:grid-cols-[1fr_320px]">
      <div className="space-y-6">
        <Card>
          <CardHeader>
            <CardTitle className="text-base font-semibold text-foreground">
              Model Parameters
            </CardTitle>
          </CardHeader>
          <CardContent className="grid gap-6 sm:grid-cols-2">
            <ParameterField
              label="Volatility"
              description="Controls the magnitude of price fluctuations"
              value={parameters.volatility}
              onChange={(v) => updateParameter("volatility", v)}
              min={0.5}
              max={2.0}
              step={0.01}
            />
            <ParameterField
              label="Trend"
              description="Market direction bias (-1 bearish, 0 neutral, +1 bullish)"
              value={parameters.trend}
              onChange={(v) => updateParameter("trend", v)}
              min={-1}
              max={1}
              step={0.01}
            />
            <ParameterField
              label="Fat Tails"
              description="Probability of extreme price movements"
              value={parameters.fatTails}
              onChange={(v) => updateParameter("fatTails", v)}
              min={0.5}
              max={2.0}
              step={0.01}
            />
            <ParameterField
              label="V Momentum"
              description="Volatility momentum / persistence"
              value={parameters.momentum}
              onChange={(v) => updateParameter("momentum", v)}
              min={0.0}
              max={1.0}
              step={0.01}
            />
            <ParameterField
              label="Time Horizon"
              description="Number of trading days to generate"
              value={parameters.timeHorizon}
              onChange={(v) => updateParameter("timeHorizon", v)}
              min={60}
              max={500}
              step={1}
            />
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="text-base font-semibold text-foreground">
              Input Time Series
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              <p className="text-sm text-muted-foreground">
                Upload a CSV file with OHLCV headers (Open, High, Low, Close, Volume)
              </p>

              {parameters.inputFile ? (
                <div className="flex items-center justify-between rounded-lg border border-border bg-secondary/50 px-4 py-3">
                  <div className="flex items-center gap-3">
                    <FileText className="h-5 w-5 text-primary" />
                    <div>
                      <p className="text-sm font-medium text-foreground">
                        {parameters.inputFile.name}
                      </p>
                      <p className="text-xs text-muted-foreground">
                        {(parameters.inputFile.size / 1024).toFixed(1)} KB
                      </p>
                    </div>
                  </div>
                  <Button
                    variant="ghost"
                    size="icon"
                    onClick={handleRemoveFile}
                    className="text-muted-foreground hover:text-destructive"
                  >
                    <X className="h-4 w-4" />
                  </Button>
                </div>
              ) : (
                <label className="flex cursor-pointer flex-col items-center gap-2 rounded-lg border-2 border-dashed border-border px-6 py-8 transition-colors hover:border-primary/50 hover:bg-secondary/30">
                  <Upload className="h-8 w-8 text-muted-foreground" />
                  <span className="text-sm font-medium text-foreground">
                    Click to upload CSV
                  </span>
                  <span className="text-xs text-muted-foreground">
                    .csv files only
                  </span>
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept=".csv"
                    onChange={handleFileUpload}
                    className="hidden"
                  />
                </label>
              )}

              {fileError && (
                <div className="flex items-start gap-2 text-sm text-destructive">
                  <AlertCircle className="h-4 w-4 mt-0.5 shrink-0" />
                  <span>{fileError}</span>
                </div>
              )}
            </div>
          </CardContent>
        </Card>
      </div>

      <div className="lg:block">
        <Card className="sticky top-20 border-border bg-card">
          <CardHeader className="pb-3 border-b border-border">
            <CardTitle className="text-base font-semibold text-foreground">
              Configuration Summary
            </CardTitle>
          </CardHeader>
          <CardContent className="pt-4 space-y-4">
            <div className="space-y-2">
              {[
                { label: "Volatility", value: parameters.volatility.toFixed(2) },
                { label: "Trend", value: parameters.trend.toFixed(2) },
                { label: "Fat Tails", value: parameters.fatTails.toFixed(2) },
                { label: "V Momentum", value: parameters.momentum.toFixed(2) },
                { label: "Time Horizon", value: `${parameters.timeHorizon} days` },
                { label: "Time Series", value: parameters.inputFile?.name ?? "Not uploaded" },
              ].map((item) => (
                <div key={item.label} className="flex justify-between text-sm">
                  <span className="text-muted-foreground">{item.label}</span>
                  <span className="font-medium text-foreground truncate max-w-[160px]">
                    {item.value}
                  </span>
                </div>
              ))}
            </div>

            <div className="flex flex-col gap-2 pt-2 border-t border-border">
              {jobStatus && jobStatus !== "completed" && jobStatus !== "failed" && (
                <div className="flex items-center gap-2 text-sm text-muted-foreground">
                  <Loader2 className="h-3.5 w-3.5 animate-spin" />
                  <span className="capitalize">{jobStatus}…</span>
                </div>
              )}
              {jobStatus === "completed" && (
                <div className="flex items-center gap-2 text-sm text-green-600 dark:text-green-400">
                  <CheckCircle2 className="h-3.5 w-3.5" />
                  <span>Generation complete</span>
                </div>
              )}
              {generateError && (
                <div className="flex items-start gap-2 text-sm text-destructive">
                  <AlertCircle className="h-3.5 w-3.5 mt-0.5 shrink-0" />
                  <span>{generateError}</span>
                </div>
              )}
              <Button
                onClick={handleGenerate}
                disabled={isGenerating}
                className="w-full gap-2"
                size="lg"
              >
                {isGenerating ? (
                  <>
                    <Loader2 className="h-4 w-4 animate-spin" />
                    Generating…
                  </>
                ) : (
                  <>
                    <Sparkles className="h-4 w-4" />
                    Generate Data
                  </>
                )}
              </Button>
              {downloadUrl && (
                <Button
                  asChild
                  variant="secondary"
                  className="w-full gap-2"
                  size="sm"
                >
                  <a href={downloadUrl} download>
                    <Download className="h-4 w-4" />
                    Download CSV
                  </a>
                </Button>
              )}
              <Button
                variant="outline"
                onClick={handleReset}
                className="w-full gap-2 bg-transparent"
                size="sm"
              >
                <RotateCcw className="h-4 w-4" />
                Reset
              </Button>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
