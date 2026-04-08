"use client"

import { useCallback, useEffect, useRef, useState } from "react"
import {
  Loader2,
  Upload,
  RefreshCw,
  FileText,
  X,
  AlertCircle,
} from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { type UploadedModelRow, type UploadedModelsListResponse } from "@/lib/types"
import { fetchTrainingGet } from "@/lib/training-fetch"
import { useToast } from "@/components/ui/use-toast"

function formatBytes(n: number | null) {
  if (n == null) return "—"
  if (n < 1024) return `${n} B`
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`
  return `${(n / (1024 * 1024)).toFixed(1)} MB`
}

function formatDate(iso: string) {
  return new Intl.DateTimeFormat("en-SG", {
    dateStyle: "medium",
    timeStyle: "short",
  }).format(new Date(iso))
}

interface Props {
  /** `unified`: no card title / section labels — for Active model tab alongside runs. */
  variant?: "default" | "unified"
}

export function ModelUploadPanel({ variant = "default" }: Props) {
  const { toast } = useToast()
  const fileInputRef = useRef<HTMLInputElement>(null)
  const [uploads, setUploads] = useState<UploadedModelRow[]>([])
  const [loading, setLoading] = useState(true)
  const [uploading, setUploading] = useState(false)
  const [file, setFile] = useState<File | null>(null)
  const [fileError, setFileError] = useState<string | null>(null)

  const fetchUploads = useCallback(async () => {
    setLoading(true)
    try {
      const res = await fetchTrainingGet("/api/training/uploads")
      if (!res.ok) throw new Error(`Server returned ${res.status}`)
      const data: UploadedModelsListResponse = await res.json()
      setUploads(data.uploads ?? [])
    } catch {
      toast({
        title: "Error",
        description: "Could not load uploaded models.",
        variant: "destructive",
      })
    } finally {
      setLoading(false)
    }
  }, [toast])

  useEffect(() => {
    fetchUploads()
  }, [fetchUploads])

  function handleFileChange(e: React.ChangeEvent<HTMLInputElement>) {
    setFileError(null)
    const f = e.target.files?.[0]
    if (!f) {
      setFile(null)
      return
    }
    if (!f.name.toLowerCase().endsWith(".pkl")) {
      setFile(null)
      setFileError("Please upload a .pkl file")
      if (fileInputRef.current) fileInputRef.current.value = ""
      return
    }
    setFile(f)
  }

  function handleRemoveFile() {
    setFile(null)
    setFileError(null)
    if (fileInputRef.current) fileInputRef.current.value = ""
  }

  async function handleUploadToS3() {
    if (!file) {
      toast({
        title: "Choose a file",
        description: "Select a .pkl file to upload.",
        variant: "destructive",
      })
      return
    }

    setUploading(true)
    try {
      const fd = new FormData()
      fd.append("file", file)
      const res = await fetch("/api/training/models/upload", {
        method: "POST",
        body: fd,
      })
      const data = await res.json().catch(() => ({}))
      if (!res.ok) {
        throw new Error(typeof data.detail === "string" ? data.detail : "Upload failed")
      }
      toast({
        title: "Uploaded",
        description: `Saved as ${data.model_name ?? "new model"} in S3.`,
      })
      handleRemoveFile()
      await fetchUploads()
    } catch (err) {
      toast({
        title: "Upload failed",
        description: err instanceof Error ? err.message : "Unknown error",
        variant: "destructive",
      })
    } finally {
      setUploading(false)
    }
  }

  const uploadIntro =
    variant === "unified" ? (
      <p className="text-sm text-muted-foreground">
        <span className="font-mono text-xs">.pkl</span> files use the same S3 naming as trained artifacts (
        <span className="font-mono text-xs">rf_delta_YYYYMMDD_HHMMSS.pkl</span>).
      </p>
    ) : (
      <p className="text-sm text-muted-foreground">
        Stored in S3 as{" "}
        <span className="font-mono text-xs">rf_delta_YYYYMMDD_HHMMSS.pkl</span> (same naming as trained
        models). Use the <span className="font-medium text-foreground">Active model</span> tab to set which
        file is live.
      </p>
    )

  const uploadBody = (
    <div className="space-y-3">
      {uploadIntro}

            {file ? (
              <div className="flex items-center justify-between rounded-lg border border-border bg-secondary/50 px-4 py-3">
                <div className="flex items-center gap-3">
                  <FileText className="h-5 w-5 text-primary" />
                  <div>
                    <p className="text-sm font-medium text-foreground">{file.name}</p>
                    <p className="text-xs text-muted-foreground">{formatBytes(file.size)}</p>
                  </div>
                </div>
                <Button
                  type="button"
                  variant="ghost"
                  size="icon"
                  onClick={handleRemoveFile}
                  disabled={uploading}
                  className="text-muted-foreground hover:text-destructive"
                >
                  <X className="h-4 w-4" />
                </Button>
              </div>
            ) : (
              <label
                className="flex cursor-pointer flex-col items-center gap-2 rounded-lg border-2 border-dashed border-border px-6 py-8 transition-colors hover:border-primary/50 hover:bg-secondary/30"
                aria-label="Click to upload PKL"
              >
                <Upload className="h-8 w-8 text-muted-foreground" />
                <span className="text-sm font-medium text-foreground">Click to upload PKL</span>
                <span className="text-xs text-muted-foreground">.pkl files only</span>
                <input
                  ref={fileInputRef}
                  type="file"
                  accept=".pkl,application/octet-stream"
                  onChange={handleFileChange}
                  disabled={uploading}
                  className="hidden"
                />
              </label>
            )}

            {fileError && (
              <div className="flex items-start gap-2 text-sm text-destructive">
                <AlertCircle className="mt-0.5 h-4 w-4 shrink-0" />
                <span>{fileError}</span>
              </div>
            )}

            {file && (
              <Button
                type="button"
                disabled={uploading}
                className="gap-2"
                onClick={handleUploadToS3}
              >
                {uploading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Upload className="h-4 w-4" />}
                {uploading ? "Uploading…" : "Upload"}
              </Button>
            )}
    </div>
  )

  return (
    <div className="space-y-6">
      {variant === "unified" ? (
        <div className="rounded-2xl border border-border bg-card shadow-sm p-5">{uploadBody}</div>
      ) : (
        <Card>
          <CardHeader>
            <CardTitle className="text-base font-semibold text-foreground">Upload model file</CardTitle>
          </CardHeader>
          <CardContent>{uploadBody}</CardContent>
        </Card>
      )}

      <div
        className={
          variant === "unified" ? "flex justify-end" : "flex items-center justify-between"
        }
      >
        {variant !== "unified" && (
          <h3 className="text-sm font-semibold text-foreground">
            Uploaded models
            {!loading && (
              <span className="ml-2 text-muted-foreground font-normal">({uploads.length} total)</span>
            )}
          </h3>
        )}
        <Button
          type="button"
          variant="outline"
          size="sm"
          className="h-8 gap-1.5"
          onClick={() => fetchUploads()}
          disabled={loading}
        >
          <RefreshCw className={`h-3.5 w-3.5 ${loading ? "animate-spin" : ""}`} />
          Refresh
        </Button>
      </div>

      <div className="rounded-md border border-border bg-card overflow-hidden">
        {loading ? (
          <div className="flex justify-center py-16">
            <Loader2 className="h-8 w-8 animate-spin text-primary" />
          </div>
        ) : uploads.length === 0 ? (
          <p className="py-12 text-center text-sm text-muted-foreground">
            No uploaded models yet. Upload a .pkl file above.
          </p>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-border bg-muted/40">
                  <th className="px-4 py-3 text-left font-medium text-muted-foreground">Model name</th>
                  <th className="px-4 py-3 text-left font-medium text-muted-foreground">Size</th>
                  <th className="px-4 py-3 text-left font-medium text-muted-foreground">Uploaded</th>
                </tr>
              </thead>
              <tbody>
                {uploads.map((row) => (
                  <tr key={row.id} className="border-b border-border">
                    <td className="px-4 py-3 font-mono text-xs">{row.model_name}</td>
                    <td className="px-4 py-3 text-muted-foreground tabular-nums">
                      {formatBytes(row.file_size_bytes)}
                    </td>
                    <td className="px-4 py-3 text-muted-foreground whitespace-nowrap text-xs">
                      {formatDate(row.created_at)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  )
}
