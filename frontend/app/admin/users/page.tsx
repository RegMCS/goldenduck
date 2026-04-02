"use client"

import React, { useEffect, useState } from "react"
import { useAuth, AuthUser } from "@/components/auth-provider"
import { useRouter } from "next/navigation"
import { Header } from "@/components/header"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { Switch } from "@/components/ui/switch"
import { useToast } from "@/components/ui/use-toast"
import { Loader2, Shield } from "lucide-react"

interface User extends AuthUser {
  is_admin?: boolean
}

export default function AdminUsersPage() {
  const { user, loading: authLoading } = useAuth()
  const router = useRouter()
  const { toast } = useToast()
  
  const [users, setUsers] = useState<User[]>([])
  const [loading, setLoading] = useState(true)
  const [updating, setUpdating] = useState<Set<string>>(new Set())

  useEffect(() => {
    // If auth loaded and user not logged in, redirect
    if (!authLoading && !user) {
      router.push("/login")
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
    
    if (user) {
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
          <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
        </div>
      </div>
    )
  }

  if (!user) return null

  return (
    <div className="min-h-screen bg-background flex flex-col">
      <Header />
      <main className="container mx-auto max-w-5xl px-4 py-8">
        <div className="mb-6 flex items-center gap-3">
          <div className="p-2 bg-primary/10 rounded-lg">
            <Shield className="w-6 h-6 text-primary" />
          </div>
          <div>
            <h1 className="text-2xl font-bold tracking-tight text-foreground sm:text-3xl">
              User Access Management
            </h1>
            <p className="text-sm text-muted-foreground sm:text-base mt-1">
              Control user permissions and administrator access across the platform.
            </p>
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
