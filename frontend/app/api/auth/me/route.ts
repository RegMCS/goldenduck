import { NextRequest, NextResponse } from "next/server"
import { BACKEND_SERVICE_URL } from "@/lib/server-constants"
import { getAccessTokenFromRequest, clearAuthCookieOnResponse } from "@/lib/server-auth"

export async function GET(request: NextRequest) {
  const token = getAccessTokenFromRequest(request)
  const isDev = process.env.NODE_ENV === "development"

  if (!token) {
    if (isDev) {
      try {
        const response = await fetch(`${BACKEND_SERVICE_URL}/api/auth/me`)
        if (response.ok) return NextResponse.json(await response.json())
      } catch {
        // backend unreachable; fall through to return no user
      }
    }
    const res = NextResponse.json({ user: null })
    clearAuthCookieOnResponse(res)
    return res
  }

  try {
    const response = await fetch(`${BACKEND_SERVICE_URL}/api/auth/me`, {
      headers: { Authorization: `Bearer ${token}` },
    })
    const data = await response.json()
    if (!response.ok) {
      const res = NextResponse.json({ user: null })
      clearAuthCookieOnResponse(res)
      return res
    }
    return NextResponse.json(data)
  } catch (error) {
    console.error("Error calling backend /api/auth/me:", error)
    const res = NextResponse.json(
      { error: "Failed to get current user" },
      { status: 500 }
    )
    clearAuthCookieOnResponse(res)
    return res
  }
}
