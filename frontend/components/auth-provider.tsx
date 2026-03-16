"use client"

import React, { createContext, useCallback, useContext, useEffect, useState } from "react"

export interface AuthUser {
  id: string
  username: string
  first_name?: string | null
  last_name?: string | null
}

interface AuthContextValue {
  user: AuthUser | null
  loading: boolean
  error: string | null
  logout: () => Promise<void>
  refresh: () => Promise<void>
  setUserFromLogin: (user: AuthUser) => void
}

const AuthContext = createContext<AuthContextValue | null>(null)

export function useAuth(): AuthContextValue {
  const ctx = useContext(AuthContext)
  if (!ctx) {
    throw new Error("useAuth must be used within an AuthProvider")
  }
  return ctx
}

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [user, setUser] = useState<AuthUser | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const fetchUser = useCallback(async () => {
    setError(null)
    try {
      const res = await fetch("/api/auth/me", { credentials: "include" })
      const data = await res.json().catch(() => ({}))
      if (res.ok && data.id != null) {
        setUser({
          id: data.id,
          username: data.username,
          first_name: data.first_name ?? null,
          last_name: data.last_name ?? null,
        })
      } else {
        setUser(null)
      }
    } catch {
      setUser(null)
      setError("Failed to load session")
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    fetchUser()
  }, [fetchUser])

  const logout = useCallback(async () => {
    try {
      await fetch("/api/auth/logout", { method: "POST", credentials: "include" })
    } finally {
      setUser(null)
    }
  }, [])

  const refresh = useCallback(async () => {
    setLoading(true)
    await fetchUser()
  }, [fetchUser])

  const setUserFromLogin = useCallback((u: AuthUser) => {
    setUser(u)
    setError(null)
  }, [])

  const value: AuthContextValue = {
    user,
    loading,
    error,
    logout,
    refresh,
    setUserFromLogin,
  }

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>
}
