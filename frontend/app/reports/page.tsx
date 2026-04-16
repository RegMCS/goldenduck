"use client"

import { useEffect, useState, useCallback } from "react"
import Link from "next/link"
import {
    Activity,
    ArrowLeft,
    RefreshCw,
    Download,
    Clock,
    CheckCircle2,
    XCircle,
    Loader2,
    BarChart3,
    Calendar,
    ChevronLeft,
    ChevronRight,
    Filter,
} from "lucide-react"
import { Header } from "@/components/header"
import { Button } from "@/components/ui/button"
import {
    Select,
    SelectContent,
    SelectItem,
    SelectTrigger,
    SelectValue,
} from "@/components/ui/select"
import {
    DropdownMenu,
    DropdownMenuContent,
    DropdownMenuItem,
    DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"
import { type JobHistoryResponse, type JobStatus } from "@/lib/types"
import { useAuth } from "@/components/auth-provider"

// ── Constants ────────────────────────────────────────────────────────────────
const PAGE_SIZE = 10

// ── Helpers ──────────────────────────────────────────────────────────────────
function formatDate(iso: string) {
    return new Intl.DateTimeFormat("en-US", {
        dateStyle: "medium",
        timeStyle: "short",
    }).format(new Date(iso))
}

function duration(start: string, end: string | null) {
    if (!end) return "-"
    const ms = new Date(end).getTime() - new Date(start).getTime()
    const secs = Math.floor(ms / 1000)
    if (secs < 60) return `${secs}s`
    const mins = Math.floor(secs / 60)
    return `${mins}m ${secs % 60}s`
}

// ── Sub-components ────────────────────────────────────────────────────────────
function StatusBadge({ status }: { status: JobStatus }) {
    const map: Record<
        JobStatus,
        { label: string; icon: React.ReactNode; className: string }
    > = {
        queued: {
            label: "Queued",
            icon: <Clock className="h-3 w-3" />,
            className:
                "bg-amber-100 text-amber-700 border-amber-200 dark:bg-amber-900/30 dark:text-amber-400 dark:border-amber-700",
        },
        running: {
            label: "Running",
            icon: <Loader2 className="h-3 w-3 animate-spin" />,
            className:
                "bg-blue-100 text-blue-700 border-blue-200 dark:bg-blue-900/30 dark:text-blue-400 dark:border-blue-700",
        },
        completed: {
            label: "Completed",
            icon: <CheckCircle2 className="h-3 w-3" />,
            className:
                "bg-emerald-100 text-emerald-700 border-emerald-200 dark:bg-emerald-900/30 dark:text-emerald-400 dark:border-emerald-700",
        },
        failed: {
            label: "Failed",
            icon: <XCircle className="h-3 w-3" />,
            className:
                "bg-red-100 text-red-700 border-red-200 dark:bg-red-900/30 dark:text-red-400 dark:border-red-700",
        },
    }
    const { label, icon, className } = map[status] ?? map.queued
    return (
        <span
            className={`inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-xs font-medium ${className}`}
        >
            {icon}
            {label}
        </span>
    )
}

function JobTypeBadge({ jobType }: { jobType: string | null }) {
    if (!jobType) return <span className="text-muted-foreground text-sm">—</span>
    const colours: Record<string, string> = {
        garch:
            "bg-violet-100 text-violet-700 border-violet-200 dark:bg-violet-900/30 dark:text-violet-400 dark:border-violet-700",
        gan: "bg-sky-100 text-sky-700 border-sky-200 dark:bg-sky-900/30 dark:text-sky-400 dark:border-sky-700",
        ddpm: "bg-fuchsia-100 text-fuchsia-700 border-fuchsia-200 dark:bg-fuchsia-900/30 dark:text-fuchsia-400 dark:border-fuchsia-700",
    }
    return (
        <span
            className={`inline-flex items-center rounded-full border px-2.5 py-1 text-xs font-semibold uppercase tracking-wide ${colours[jobType] ?? "bg-muted text-muted-foreground border-border"}`}
        >
            {jobType}
        </span>
    )
}

function EmptyState() {
    return (
        <div className="flex flex-col items-center justify-center py-24 text-center">
            <div className="mb-4 flex h-16 w-16 items-center justify-center rounded-2xl bg-muted">
                <BarChart3 className="h-8 w-8 text-muted-foreground" />
            </div>
            <h3 className="text-lg font-semibold text-foreground">No reports yet</h3>
            <p className="mt-1 max-w-xs text-sm text-muted-foreground">
                Generate your first synthetic dataset to see it appear here.
            </p>
            <Button asChild className="mt-6 gap-2">
                <Link href="/">
                    <Activity className="h-4 w-4" />
                    Generate Dataset
                </Link>
            </Button>
        </div>
    )
}

function SkeletonRow() {
    return (
        <tr className="border-b border-border animate-pulse">
            {Array.from({ length: 6 }).map((_, i) => (
                <td key={i} className="px-4 py-4">
                    <div className="h-4 w-full rounded-md bg-muted" />
                </td>
            ))}
        </tr>
    )
}

// ── Main Page ─────────────────────────────────────────────────────────────────
export default function ReportsPage() {
    const [data, setData] = useState<JobHistoryResponse | null>(null)
    const [loading, setLoading] = useState(true)
    const [error, setError] = useState<string | null>(null)
    const [statusFilter, setStatusFilter] = useState<string>("all")
    const [page, setPage] = useState(0)
    const { user } = useAuth()
    const userId = user?.id ?? ""

    const fetchHistory = useCallback(async () => {
        if (!userId) return
        setLoading(true)
        setError(null)
        try {
            const params = new URLSearchParams({
                limit: String(PAGE_SIZE),
                offset: String(page * PAGE_SIZE),
            })
            if (statusFilter !== "all") params.set("status", statusFilter)

            const res = await fetch(`/api/history/${userId}?${params.toString()}`)
            if (!res.ok) throw new Error(`Server returned ${res.status}`)
            const json: JobHistoryResponse = await res.json()
            setData(json)
        } catch (err) {
            setError(err instanceof Error ? err.message : "Unknown error")
        } finally {
            setLoading(false)
        }
    }, [statusFilter, page, userId])

    useEffect(() => {
        if (userId) fetchHistory()
    }, [fetchHistory, userId])

    const totalPages = data ? Math.ceil(data.total / PAGE_SIZE) : 0

    return (
        <div className="min-h-screen bg-background">
            <Header />

            <main className="container mx-auto max-w-7xl px-4 py-6 sm:py-8">
                {!user && (
                    <div className="rounded-lg border border-border bg-muted/30 px-6 py-8 text-center">
                        <p className="text-muted-foreground mb-2">Log in to view your reports.</p>
                        <Button asChild>
                            <Link href="/login">Log in</Link>
                        </Button>
                    </div>
                )}
                {user && (
                <>
                {/* ── Page Header ── */}
                <div className="mb-6 flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between">
                    <div>
                        <div className="mb-1 flex items-center gap-2">
                            <Button
                                variant="ghost"
                                size="sm"
                                className="h-8 gap-1.5 -ml-2 text-muted-foreground hover:text-foreground"
                                asChild
                            >
                                <Link href="/">
                                    <ArrowLeft className="h-4 w-4" />
                                    Back
                                </Link>
                            </Button>
                        </div>
                        <div className="flex items-center gap-3">
                            <div className="flex h-10 w-10 flex-shrink-0 items-center justify-center rounded-xl bg-primary/10 shadow-sm">
                                <BarChart3 className="h-5 w-5 text-primary" />
                            </div>
                            <div>
                                <h2 className="text-xl font-bold tracking-tight text-foreground sm:text-2xl">
                                    Generated Reports
                                </h2>
                                <p className="mt-0.5 text-sm text-muted-foreground">
                                    All previously submitted AI model jobs and their results.
                                </p>
                            </div>
                        </div>
                    </div>

                    <div className="flex items-center gap-2 flex-shrink-0">
                        {/* Status filter */}
                        <div className="flex items-center gap-1.5">
                            <Filter className="h-3.5 w-3.5 text-muted-foreground" />
                            <Select
                                value={statusFilter}
                                onValueChange={(v) => {
                                    setStatusFilter(v)
                                    setPage(0)
                                }}
                            >
                                <SelectTrigger className="h-8 w-[130px] text-xs" id="status-filter">
                                    <SelectValue placeholder="All statuses" />
                                </SelectTrigger>
                                <SelectContent>
                                    <SelectItem value="all">All statuses</SelectItem>
                                    <SelectItem value="completed">Completed</SelectItem>
                                    <SelectItem value="running">Running</SelectItem>
                                    <SelectItem value="queued">Queued</SelectItem>
                                    <SelectItem value="failed">Failed</SelectItem>
                                </SelectContent>
                            </Select>
                        </div>

                        <Button
                            variant="outline"
                            size="sm"
                            className="gap-1.5 h-8"
                            onClick={fetchHistory}
                            disabled={loading}
                            id="refresh-reports-btn"
                        >
                            <RefreshCw className={`h-3.5 w-3.5 ${loading ? "animate-spin" : ""}`} />
                            Refresh
                        </Button>
                    </div>
                </div>

                {/* ── Summary Strip ── */}
                {data && (
                    <div className="mb-5 grid grid-cols-2 gap-3 sm:grid-cols-4">
                        {(
                            [
                                {
                                    label: "Total",
                                    value: data.total_completed + data.total_running + data.total_failed + data.total_queued,
                                    color: "text-foreground",
                                },
                                {
                                    label: "Completed",
                                    value: data.total_completed,
                                    color: "text-emerald-600 dark:text-emerald-400",
                                },
                                {
                                    label: "Running",
                                    value: data.total_running,
                                    color: "text-blue-600 dark:text-blue-400",
                                },
                                {
                                    label: "Failed",
                                    value: data.total_failed,
                                    color: "text-red-600 dark:text-red-400",
                                },
                            ] as { label: string; value: number; color: string }[]
                        ).map((stat) => (
                            <div
                                key={stat.label}
                                className="rounded-xl border border-border bg-card px-4 py-3 shadow-sm"
                            >
                                <p className="text-xs text-muted-foreground">{stat.label}</p>
                                <p className={`mt-0.5 text-2xl font-bold tabular-nums ${stat.color}`}>
                                    {stat.value}
                                </p>
                            </div>
                        ))}
                    </div>
                )}

                {/* ── Table Card ── */}
                <div className="rounded-2xl border border-border bg-card shadow-sm overflow-hidden">
                    {error && (
                        <div className="flex items-center gap-2 border-b border-border bg-destructive/10 px-6 py-3 text-sm text-destructive">
                            <XCircle className="h-4 w-4 flex-shrink-0" />
                            <span>Failed to load reports: {error}</span>
                            <Button
                                variant="ghost"
                                size="sm"
                                className="ml-auto h-7 text-xs"
                                onClick={fetchHistory}
                            >
                                Retry
                            </Button>
                        </div>
                    )}

                    <div className="overflow-x-auto">
                        <table className="w-full text-sm">
                            <thead>
                                <tr className="border-b border-border bg-muted/40">
                                    {["Job ID", "Model", "Status", "Requested", "Duration", "Actions"].map(
                                        (h) => (
                                            <th
                                                key={h}
                                                className="px-4 py-3 text-left text-xs font-semibold uppercase tracking-wider text-muted-foreground"
                                            >
                                                {h}
                                            </th>
                                        )
                                    )}
                                </tr>
                            </thead>
                            <tbody>
                                {loading &&
                                    Array.from({ length: PAGE_SIZE }).map((_, i) => (
                                        <SkeletonRow key={i} />
                                    ))}

                                {!loading && data && data.jobs.length === 0 && (
                                    <tr>
                                        <td colSpan={6}>
                                            <EmptyState />
                                        </td>
                                    </tr>
                                )}

                                {!loading &&
                                    data?.jobs.map((job) => (
                                        <tr
                                            key={job.id}
                                            className="border-b border-border/60 transition-colors hover:bg-muted/30 last:border-0"
                                        >
                                            {/* Job ID */}
                                            <td className="px-4 py-3.5 font-mono text-xs text-muted-foreground">
                                                <span title={job.id}>{job.id.slice(0, 8)}…</span>
                                            </td>

                                            {/* Model type */}
                                            <td className="px-4 py-3.5">
                                                <JobTypeBadge jobType={job.job_type} />
                                            </td>

                                            {/* Status */}
                                            <td className="px-4 py-3.5">
                                                <StatusBadge status={job.status} />
                                            </td>

                                            {/* Requested at */}
                                            <td className="px-4 py-3.5 text-muted-foreground whitespace-nowrap">
                                                <span className="flex items-center gap-1.5">
                                                    <Calendar className="h-3.5 w-3.5 flex-shrink-0" />
                                                    {formatDate(job.requested_at)}
                                                </span>
                                            </td>

                                            {/* Duration */}
                                            <td className="px-4 py-3.5 text-muted-foreground tabular-nums">
                                                {duration(job.requested_at, job.completed_at)}
                                            </td>

                                            {/* Actions */}
                                            <td className="px-4 py-3.5">
                                                {job.status === "completed" ? (
                                                    <div className="flex items-center gap-1">
                                                        <Link
                                                            href={`/results?jobId=${job.id}`}
                                                            className="inline-flex items-center gap-1.5 rounded-md px-2.5 py-1 text-xs font-medium text-primary hover:bg-muted transition-colors"
                                                        >
                                                            <BarChart3 className="h-3.5 w-3.5" />
                                                            View
                                                        </Link>
                                                        <DropdownMenu>
                                                            <DropdownMenuTrigger asChild>
                                                                <button
                                                                    type="button"
                                                                    className="inline-flex items-center gap-1.5 rounded-md px-2.5 py-1 text-xs font-medium text-muted-foreground hover:bg-muted transition-colors"
                                                                >
                                                                    <Download className="h-3.5 w-3.5" />
                                                                    Download
                                                                </button>
                                                            </DropdownMenuTrigger>
                                                            <DropdownMenuContent align="end" className="w-44">
                                                                <DropdownMenuItem asChild>
                                                                    <a
                                                                        id={`download-selected-btn-${job.id}`}
                                                                        href={`/api/download/user/${userId}/${job.id}/selected-path`}
                                                                        target="_blank"
                                                                        rel="noreferrer"
                                                                    >
                                                                        Download Selected
                                                                    </a>
                                                                </DropdownMenuItem>
                                                                <DropdownMenuItem asChild>
                                                                    <a
                                                                        id={`download-btn-${job.id}`}
                                                                        href={`/api/download/user/${userId}/${job.id}`}
                                                                        target="_blank"
                                                                        rel="noreferrer"
                                                                    >
                                                                        Download All
                                                                    </a>
                                                                </DropdownMenuItem>
                                                            </DropdownMenuContent>
                                                        </DropdownMenu>
                                                    </div>
                                                ) : (
                                                    <span className="text-muted-foreground text-xs">—</span>
                                                )}
                                            </td>
                                        </tr>
                                    ))}
                            </tbody>
                        </table>
                    </div>

                    {/* ── Pagination ── */}
                    {totalPages > 1 && (
                        <div className="flex items-center justify-between border-t border-border px-4 py-3">
                            <p className="text-xs text-muted-foreground">
                                Page {page + 1} of {totalPages} · {data?.total} total
                            </p>
                            <div className="flex items-center gap-1">
                                <Button
                                    variant="outline"
                                    size="icon"
                                    className="h-7 w-7"
                                    disabled={page === 0}
                                    onClick={() => setPage((p) => p - 1)}
                                    id="prev-page-btn"
                                >
                                    <ChevronLeft className="h-4 w-4" />
                                </Button>
                                <Button
                                    variant="outline"
                                    size="icon"
                                    className="h-7 w-7"
                                    disabled={page >= totalPages - 1}
                                    onClick={() => setPage((p) => p + 1)}
                                    id="next-page-btn"
                                >
                                    <ChevronRight className="h-4 w-4" />
                                </Button>
                            </div>
                        </div>
                    )}
                </div>
                </>
                )}
            </main>
        </div>
    )
}
