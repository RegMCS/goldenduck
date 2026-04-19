"use client"

import React, { useState, useRef, useEffect } from "react"
import { useRouter } from "next/navigation"
import { Upload, X, FileText, Sparkles, RotateCcw, Download, AlertCircle, CheckCircle2, Loader2, TriangleAlert, ChevronUp, ChevronDown } from "lucide-react"
import { Input } from "@/components/ui/input"
import { Button } from "@/components/ui/button"
import { Label } from "@/components/ui/label"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { InfoTooltip } from "@/components/info-tooltip"
import { TimingEstimate } from "@/components/timing-estimate"
import { ToggleParameter } from "@/components/toggle-parameter"
import {
  type MarketParameters,
  defaultParameters,
  REQUIRED_CSV_HEADERS,
} from "@/lib/types"
import { useAuth } from "@/components/auth-provider"
import Link from "next/link"

const MIN_HORIZON_DAYS = 500
const MAX_HORIZON_DAYS = 2600
const MIN_CSV_ROWS = 500
const TWEAKABLE_KNOBS = [
  { key: "volatility", label: "Volatility" },
  { key: "trend", label: "Trend" },
  { key: "fatTails", label: "Fat Tails" },
  { key: "momentum", label: "Momentum" },
] as const
type TweakedKnobKey = (typeof TWEAKABLE_KNOBS)[number]["key"]
type KnobBaseline = Pick<MarketParameters, TweakedKnobKey>
type PresetMode = "bull" | "flash" | null

type JobStatus = "queued" | "running" | "completed" | "failed"

export function getTweakedKnobLabels(
  parameters: MarketParameters,
  baseline: KnobBaseline = defaultParameters
): string[] {
  return TWEAKABLE_KNOBS.filter(
    ({ key }) => parameters[key] !== baseline[key]
  ).map(({ label }) => label)
}


interface ParameterFieldProps {
  label: string
  description: string
  tooltip?: string | string[]
  value: number
  onChange: (value: number) => void
  min: number
  max: number
  step: number
}

function ParameterField({ label, description, tooltip, value, onChange, min, max, step }: ParameterFieldProps) {
  const [localValue, setLocalValue] = React.useState(String(value))

  // Sync when the parent resets the value externally
  React.useEffect(() => {
    setLocalValue(String(value))
  }, [value])

  const commit = () => {
    const parsed = parseFloat(localValue)
    if (!isNaN(parsed)) {
      const clamped = Math.min(max, Math.max(min, parsed))
      onChange(clamped)
      setLocalValue(String(clamped))
    } else {
      setLocalValue(String(value))
    }
  }

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-1.5">
          <Label className="text-sm font-medium text-foreground">{label}</Label>
          {tooltip && <InfoTooltip content={tooltip} />}
        </div>
        <span className="text-xs text-muted-foreground">{min} – {max}</span>
      </div>
      <div className="flex">
        <Input
          type="text"
          inputMode="decimal"
          aria-label={label}
          value={localValue}
          onChange={(e) => setLocalValue(e.target.value)}
          onBlur={commit}
          onKeyDown={(e) => {
            if (e.key === "Enter") {
              commit()
            } else if (e.key === "ArrowUp" || e.key === "ArrowDown") {
              e.preventDefault()
              const current = parseFloat(localValue)
              const base = isNaN(current) ? parseFloat(String(value)) : current
              const next = Math.min(max, Math.max(min, parseFloat((base + (e.key === "ArrowUp" ? step : -step)).toFixed(10))))
              setLocalValue(String(next))
              onChange(next)
            }
          }}
          className="rounded-r-none"
        />
        <div className="flex flex-col border border-l-0 border-input rounded-r-md overflow-hidden">
          <button
            type="button"
            tabIndex={-1}
            onMouseDown={(e) => {
              e.preventDefault()
              const current = parseFloat(localValue)
              const base = isNaN(current) ? parseFloat(String(value)) : current
              const next = Math.min(max, parseFloat((base + step).toFixed(10)))
              setLocalValue(String(next))
              onChange(next)
            }}
            className="flex-1 flex items-center justify-center px-1.5 text-muted-foreground hover:text-foreground hover:bg-muted transition-colors"
          >
            <ChevronUp className="h-3 w-3" />
          </button>
          <div className="h-px bg-input" />
          <button
            type="button"
            tabIndex={-1}
            onMouseDown={(e) => {
              e.preventDefault()
              const current = parseFloat(localValue)
              const base = isNaN(current) ? parseFloat(String(value)) : current
              const next = Math.max(min, parseFloat((base - step).toFixed(10)))
              setLocalValue(String(next))
              onChange(next)
            }}
            className="flex-1 flex items-center justify-center px-1.5 text-muted-foreground hover:text-foreground hover:bg-muted transition-colors"
          >
            <ChevronDown className="h-3 w-3" />
          </button>
        </div>
      </div>
      <p className="text-xs text-muted-foreground">{description}</p>
    </div>
  )
}

const countCsvRows = (text: string): number => {
  const lines = text
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter((line) => line.length > 0)
  return Math.max(lines.length - 1, 0)
}

interface TradeoffRule {
  condition: (p: MarketParameters) => boolean
  message: string
}

const TRADEOFF_RULES: TradeoffRule[] = [
  // {
  //   condition: (p) => p.timeHorizon < MIN_HORIZON_DAYS,
  //   message:
  //     `A time horizon below ${MIN_HORIZON_DAYS} days reduces the amount of data available for the model to learn from. We'll do our best, but results may be less statistically reliable — a longer horizon will produce more accurate synthetic data.`,
  // },
  // to check
  {
    condition: (p) => p.momentum > 0.7,
    message:
      "Setting Momentum > 0.7 will increase volatility as a natural side effect of return persistence, though fat tails may simultaneously decrease as extreme random spikes are smoothed out by the persistent trend.",
  },
  {
    condition: (p) => p.momentum < 0.2,
    message:
      "Setting Momentum < 0.3 will increase volatility as a natural side effect of frequent return reversals, while also reducing fat tails as the erratic but bounded fluctuations replace the sustained extreme moves that drive excess kurtosis.",
  },
  {
    condition: (p) => p.fatTails > 1.5 && p.momentum > 0.7,
    message:
      "Combining Fat Tails > 1.5 with Momentum > 0.7 will cause kurtosis to hit the model ceiling (~8.0) — expect extreme tail behaviour.",
  },
  {
    condition: (p) => p.trend > 0.7 || p.trend < -0.7,
    message:
      "Setting Trend > 0.7 or < −0.7 will naturally dampen volatility by imposing more structure on the output distribution.",
  },
  // {
  //   condition: (p) => Math.abs(p.trend) > 0,
  //   message:
  //     `Non-zero Trend is easier to observe over longer horizons. Recommended: at least 1000 days (preferably 2000+ days).`,
  // },
]

let defaultCsvLoadPromise: Promise<File | null> | null = null

const loadDefaultCsvFile = async (): Promise<File | null> => {
  try {
    const response = await fetch("/api/default-csv")
    if (!response.ok) {
      return null
    }

    const contentType = response.headers.get("content-type")
    let blob: Blob

    if (contentType?.includes("application/json")) {
      const data = await response.json()
      if (!data.url) return null

      const csvResponse = await fetch(data.url)
      if (!csvResponse.ok) return null
      blob = await csvResponse.blob()
    } else {
      blob = await response.blob()
    }

    const file = new File([blob], "AAPL_real.csv", { type: "text/csv" })

    const validation = await new Promise<{ validHeaders: boolean; rowCount: number }>((resolve) => {
      const reader = new FileReader()
      reader.onload = (e) => {
        const text = e.target?.result as string
        const lines = text
          .split(/\r?\n/)
          .map((l) => l.trim())
          .filter((l) => l.length > 0)
        const firstLine = (lines[0] ?? "").toLowerCase().trim()
        const headers = firstLine.split(",").map((h) => h.trim())
        const hasAllHeaders = REQUIRED_CSV_HEADERS.every((required) =>
          headers.includes(required)
        )
        const rowCount = countCsvRows(text)
        resolve({ validHeaders: hasAllHeaders, rowCount })
      }
      reader.onerror = () => resolve({ validHeaders: false, rowCount: 0 })
      reader.readAsText(file)
    })

    if (!validation.validHeaders || validation.rowCount < MIN_CSV_ROWS) {
      return null
    }

    return file
  } catch {
    return null
  }
}

export function ParameterizationForm() {
  const router = useRouter()
  const { user, logout } = useAuth()
  const [parameters, setParameters] = useState<MarketParameters>(defaultParameters)
  const [knobBaseline, setKnobBaseline] = useState<KnobBaseline>(defaultParameters)
  const [presetMode, setPresetMode] = useState<PresetMode>(null)
  const [isGenerating, setIsGenerating] = useState(false)
  const [fileError, setFileError] = useState<string | null>(null)
  const [jobStatus, setJobStatus] = useState<JobStatus | null>(null)
  const [downloadUrl, setDownloadUrl] = useState<string | null>(null)
  const [generateError, setGenerateError] = useState<string | null>(null)
  const [useSkewShocks, setUseSkewShocks] = useState(true)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const pollIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null)

  // Auto-load AAPL_real.csv on component mount
  useEffect(() => {
    let isMounted = true
    if (!defaultCsvLoadPromise) {
      defaultCsvLoadPromise = loadDefaultCsvFile()
    }

    defaultCsvLoadPromise.then((file) => {
      if (!isMounted || !file) return
      setParameters((prev) => ({
        ...prev,
        inputFile: prev.inputFile ?? file,
      }))
      setFileError(null)
    })

    return () => {
      isMounted = false
    }
  }, [])

  const activeWarnings = TRADEOFF_RULES.filter((r) => r.condition(parameters)).map((r) => r.message)
  const tweakedKnobs = getTweakedKnobLabels(parameters, knobBaseline)
  const knobLimitError =
    presetMode === null && tweakedKnobs.length > 1
      ? `You can only change one knob at a time. Reset these to default first: ${tweakedKnobs.join(", ")}.`
      : null

  const isTweakableKnobKey = (key: keyof MarketParameters): key is TweakedKnobKey =>
    TWEAKABLE_KNOBS.some((knob) => knob.key === key)


  const updateParameter = <K extends keyof MarketParameters>(
    key: K,
    value: MarketParameters[K]
  ) => {
    if (isTweakableKnobKey(key) && presetMode !== null) {
      setPresetMode(null)
      setKnobBaseline(defaultParameters)
    }
    setGenerateError(null)
    setParameters((prev) => ({ ...prev, [key]: value }))
  }

  const applyBullRunPreset = () => {
    const presetKnobs: KnobBaseline = {
      volatility: 0.5,
      trend: 0.7,
      fatTails: 0.6,
      momentum: 0.85,
    }
    setGenerateError(null)
    setPresetMode("bull")
    setKnobBaseline(presetKnobs)
    setParameters((prev) => ({
      ...prev,
      ...presetKnobs,
    }))
  }

  const applyFlashCrashPreset = () => {
    const presetKnobs: KnobBaseline = {
      volatility: 1.5,
      trend: -0.2,
      fatTails: 2.0,
      momentum: 0.5,
    }
    setGenerateError(null)
    setPresetMode("flash")
    setKnobBaseline(presetKnobs)
    setParameters((prev) => ({
      ...prev,
      ...presetKnobs,
    }))
  }

  const validateCsvStructure = (
    file: File
  ): Promise<{ validHeaders: boolean; rowCount: number }> => {
    return new Promise((resolve) => {
      const reader = new FileReader()
      reader.onload = (e) => {
        const text = e.target?.result as string
        const lines = text
          .split(/\r?\n/)
          .map((l) => l.trim())
          .filter((l) => l.length > 0)
        const firstLine = (lines[0] ?? "").toLowerCase().trim()
        const headers = firstLine.split(",").map((h) => h.trim())
        const hasAllHeaders = REQUIRED_CSV_HEADERS.every((required) =>
          headers.includes(required)
        )
        const rowCount = countCsvRows(text)
        resolve({ validHeaders: hasAllHeaders, rowCount })
      }
      reader.onerror = () => resolve({ validHeaders: false, rowCount: 0 })
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

    const { validHeaders, rowCount } = await validateCsvStructure(file)
    if (!validHeaders) {
      setFileError("CSV must have OHLCV headers: Open, High, Low, Close, Volume")
      updateParameter("inputFile", null)
      return
    }

    if (rowCount < MIN_CSV_ROWS) {
      setFileError(`CSV must contain at least ${MIN_CSV_ROWS} data rows (found ${rowCount}).`)
      updateParameter("inputFile", null)
      return
    }

    setFileError(null)
    updateParameter("inputFile", file)
    updateParameter("timeHorizon", Math.min(MAX_HORIZON_DAYS, rowCount))
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

    if (knobLimitError) {
      setGenerateError(knobLimitError)
      return
    }

    setGenerateError(null)

    setIsGenerating(true)
    setJobStatus(null)
    setDownloadUrl(null)
    setGenerateError(null)

    const userId = user?.id
    if (!userId) return

    try {
      // Read CSV file content
      const csvContent = await new Promise<string>((resolve, reject) => {
        const reader = new FileReader()
        reader.onload = (e) => resolve(e.target?.result as string)
        reader.onerror = (e) => reject(e)
        reader.readAsText(parameters.inputFile!)
      })

      const res = await fetch(`/api/generate/user/${userId}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          csv_data: csvContent,
          horizon: parameters.timeHorizon,
          desired_volatility: parameters.volatility,
          desired_trend: parameters.trend,
          desired_fat_tails: parameters.fatTails,
          desired_momentum: parameters.momentum,
          use_skew_shocks: useSkewShocks,
          force_skewt_distribution: useSkewShocks,
        }),
      })

      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: "Failed to submit job" }))
        if (res.status === 401) await logout()
        throw new Error(err.detail || "Failed to submit job")
      }

      const data = await res.json()
      setJobStatus("queued")

      pollIntervalRef.current = setInterval(async () => {
        try {
          const statusRes = await fetch(
            `/api/status/user/${userId}/${data.job_id}`
          )
          const statusData = await statusRes.json()
          setJobStatus(statusData.status as JobStatus)

          if (statusData.status === "completed") {
            clearInterval(pollIntervalRef.current!)
            setDownloadUrl(`/api/download/user/${userId}/${data.job_id}`)
            setIsGenerating(false)
            localStorage.setItem(
              `goldenduck_job_params_${data.job_id}`,
              JSON.stringify({
                volatility: parameters.volatility,
                trend: parameters.trend,
                fatTails: parameters.fatTails,
                momentum: parameters.momentum,
                timeHorizon: parameters.timeHorizon,
                fileName: parameters.inputFile?.name ?? null,
              })
            )
            router.push(`/results?jobId=${data.job_id}`)
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
    setKnobBaseline(defaultParameters)
    setPresetMode(null)
    setFileError(null)
    setJobStatus(null)
    setDownloadUrl(null)
    setGenerateError(null)
    setUseSkewShocks(true)
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

        {activeWarnings.length > 0 && (
          <div className="rounded-lg border border-amber-500/40 bg-amber-50/60 dark:bg-amber-950/20 px-4 py-3 space-y-2">
            <div className="flex items-center gap-2">
              <TriangleAlert className="h-4 w-4 text-amber-600 dark:text-amber-400 shrink-0" />
              <span className="text-xs font-semibold text-amber-700 dark:text-amber-400 uppercase tracking-wide">
                Parameter tradeoffs detected
              </span>
            </div>
            <ul className="space-y-1.5 pl-6 list-disc">
              {activeWarnings.map((msg) => (
                <li key={msg} className="text-xs text-amber-800 dark:text-amber-300">
                  {msg}
                </li>
              ))}
            </ul>
          </div>
        )}
        
        <Card>
          <CardHeader>
            <div className="flex items-center justify-between gap-3">
              <CardTitle className="text-base font-semibold text-foreground">
                Model Parameters
              </CardTitle>
              <div className="flex items-center gap-2">
                <Button
                  type="button"
                  variant="outline"
                  size="sm"
                  className="gap-2"
                  onClick={applyBullRunPreset}
                >
                  <Sparkles className="h-3.5 w-3.5" />
                  Bull Run Preset
                </Button>
                <Button
                  type="button"
                  variant="outline"
                  size="sm"
                  className="gap-2"
                  onClick={applyFlashCrashPreset}
                >
                  <Sparkles className="h-3.5 w-3.5" />
                  Flash Crash Preset
                </Button>
              </div>
            </div>
          </CardHeader>
          <CardContent className="grid gap-6 sm:grid-cols-2">
            <ParameterField
              label="Volatility"
              description="Controls the magnitude of price fluctuations"
              tooltip={[
                "Scales overall price swings.",
                "1.0 = historical level",
                "2.0 = twice as volatile",
              ]}
              value={parameters.volatility}
              onChange={(v) => updateParameter("volatility", v)}
              min={0.5}
              max={2.0}
              step={0.01}
            />
            <ParameterField
              label="Trend"
              description="Market direction bias (-1 bearish, 0 neutral, +1 bullish)"
              tooltip={[
                "Controls the overall market direction.",
                "-1.0 = strong downtrend (~−50% annual drift)",
                "-0.5 = mild downtrend (~−25% annual drift)",
                "0.0 = neutral (historical level)",
                "+0.5 = mild uptrend (~+25% annual drift)",
                "+1.0 = strong uptrend (~+50% annual drift)",
              ]}
              value={parameters.trend}
              onChange={(v) => updateParameter("trend", v)}
              min={-1}
              max={1}
              step={0.01}
            />
            <ParameterField
              label="Fat Tails"
              description="Probability of extreme price movements"
              tooltip={[
                "Controls tail heaviness (excess kurtosis).",
                "0.5 = lower kurtosis, fewer extremes",
                "1.0 = historical kurtosis baseline",
                "2.0 = higher kurtosis, more extreme moves",
              ]}
              value={parameters.fatTails}
              onChange={(v) => updateParameter("fatTails", v)}
              min={0.5}
              max={2.0}
              step={0.01}
            />
            <ParameterField
              label="Momentum"
              description="Volatility momentum / persistence"
              tooltip={[
                "Controls how much past volatility influences the next period.",
                "0.0 = no persistence (each day is independent)",
                "0.5 = moderate persistence (historical level)",
                "0.7 = strong persistence (trends carry forward noticeably)",
                "1.0 = maximum persistence (highly trending behaviour)",
              ]}
              value={parameters.momentum}
              onChange={(v) => updateParameter("momentum", v)}
              min={0.0}
              max={1.0}
              step={0.01}
            />
            <ParameterField
              label="Time Horizon"
              description={`Number of trading days to generate. Use longer horizons to clearly observe non-zero trend effects.`}
              value={parameters.timeHorizon}
              onChange={(v) => updateParameter("timeHorizon", v)}
              min={MIN_HORIZON_DAYS}
              max={MAX_HORIZON_DAYS}
              step={1}
            />
            <ToggleParameter
              label="Skew Shocks"
              description="Temporarily enable asymmetric return shocks for A/B testing."
              checked={useSkewShocks}
              onChange={setUseSkewShocks}
              className="sm:col-span-2"
            />
          </CardContent>
        </Card>

        {knobLimitError && (
          <div className="rounded-lg border border-destructive/40 bg-destructive/5 px-4 py-3">
            <div className="flex items-start gap-2 text-sm text-destructive">
              <AlertCircle className="mt-0.5 h-4 w-4 shrink-0" />
              <span>{knobLimitError}</span>
            </div>
          </div>
        )}
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
                { label: "Momentum", value: parameters.momentum.toFixed(2) },
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

            <TimingEstimate parameters={parameters} />

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
              {!user ? (
                <p className="text-sm text-muted-foreground text-center py-2">
                  <Link href="/login" className="font-medium text-primary hover:underline">
                    Log in
                  </Link>
                  {" "}to generate data.
                </p>
              ) : (
                <Button
                  onClick={handleGenerate}
                  disabled={isGenerating || !!knobLimitError}
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
              )}
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
                disabled={isGenerating}
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
