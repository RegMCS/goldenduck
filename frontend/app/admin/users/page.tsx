"use client"

import React, { useEffect, useState } from "react"
import Link from "next/link"
import { useAuth, AuthUser } from "@/components/auth-provider"
import { useRouter } from "next/navigation"
import { Header } from "@/components/header"
import { Button } from "@/components/ui/button"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { Switch } from "@/components/ui/switch"
import { useToast } from "@/components/ui/use-toast"
import { ArrowLeft, Loader2, Lock, Shield } from "lucide-react"

export default function AdminUsersPage() {
  const { user, loading: authLoading } = useAuth()
  const router = useRouter()
  const { toast } = useToast()
  
  const [users, setUsers] = useState<AuthUser[]>([])
  const [loading, setLoading] = useState(true)
  const [updating, setUpdating] = useState<Set<string>>(new Set())

  useEffect(() => {
    if (authLoading) return
    if (!user) {
      router.push("/login")
    } else if (!user.is_admin) {
      router.push("/")
    }
  }, [user, authLoading, router])

  useEffect(() => {
    const fetchUsers = async () => {
      try {
        const res = await fetch("/api/users")
        if (!res.ok) {
          throw new Error("Failed to load users")
        }
        const data = await res.json()
        setUsers(data.users || data || [])
      } catch (e) {
        toast({
          title: "Access Denied",
          description: "Could not load users. Make sure you have admin privileges.",
          variant: "destructive"
        })
      } finally {
        setLoading(false)
      }
    }
    
    if (user?.is_admin) {
      fetchUsers()
    }
  }, [user, toast])

  const toggleAdmin = async (userId: string, currentStatus: boolean) => {
    if (updating.has(userId)) return
    const newStatus = !currentStatus

    setUpdating((prev) => new Set(prev).add(userId))
    setUsers((prev) => prev.map((u) => (u.id === userId ? { ...u, is_admin: newStatus } : u)))

    try {
      const res = await fetch(`/api/users/${userId}`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ is_admin: newStatus }),
      })
      if (!res.ok) throw new Error("Failed to update user")
      toast({ title: "Success", description: "User permissions updated successfully." })
    } catch {
      setUsers((prev) => prev.map((u) => (u.id === userId ? { ...u, is_admin: currentStatus } : u)))
      toast({ title: "Error", description: "Failed to update user permissions.", variant: "destructive" })
    } finally {
      setUpdating((prev) => { const next = new Set(prev); next.delete(userId); return next })
    }
  }

  if (authLoading || loading) {
    return (
      <div className="min-h-screen bg-background flex flex-col">
        <Header />
        <div className="flex-1 flex items-center justify-center">
          <Loader2 className="h-8 w-8 animate-spin text-primary" />
        </div>
      </div>
    )
  }

  if (!user || !user.is_admin) return null

  return (
    <div className="min-h-screen bg-background flex flex-col">
      <Header />
      <main className="container mx-auto max-w-7xl px-4 py-6 sm:py-8">
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
                <Shield className="h-5 w-5 text-primary" />
              </div>
              <div>
                <h1 className="text-xl font-bold tracking-tight text-foreground sm:text-2xl">
                  User Access Management
                </h1>
                <p className="mt-0.5 text-sm text-muted-foreground">
                  Control user permissions and administrator access across the platform.
                </p>
              </div>
            </div>
          </div>

          <div className="inline-flex items-center gap-1.5 self-start rounded-full border border-amber-200 bg-amber-50 px-3 py-1 text-xs font-semibold text-amber-700 dark:border-amber-700 dark:bg-amber-900/20 dark:text-amber-400">
            <Lock className="h-3 w-3" />
            Admin Only
          </div>
        </div>

        <div className="rounded-md border bg-card">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Username</TableHead>
                <TableHead>First Name</TableHead>
                <TableHead>Last Name</TableHead>
                <TableHead className="w-[150px]">Admin Access</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {users.length === 0 ? (
                <TableRow>
                  <TableCell colSpan={4} className="h-24 text-center text-muted-foreground">
                    No users found or lack of permissions.
                  </TableCell>
                </TableRow>
              ) : (
                users.map((u) => (
                  <TableRow key={u.id}>
                    <TableCell className="font-medium">{u.username}</TableCell>
                    <TableCell>{u.first_name || "-"}</TableCell>
                    <TableCell>{u.last_name || "-"}</TableCell>
                    <TableCell>
                      <Switch
                        checked={!!u.is_admin}
                        onCheckedChange={() => toggleAdmin(u.id, !!u.is_admin)}
                        disabled={u.id === user.id}
                        aria-busy={updating.has(u.id)}
                      />
                    </TableCell>
                  </TableRow>
                ))
              )}
            </TableBody>
          </Table>
        </div>
      </main>
    </div>
  )
}
