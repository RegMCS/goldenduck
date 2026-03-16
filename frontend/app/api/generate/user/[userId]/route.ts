import { NextRequest, NextResponse } from "next/server"
import { BACKEND_SERVICE_URL } from "@/lib/server-constants"
import { getAccessTokenFromRequest, backendAuthHeaders, clearAuthCookieOnResponse } from "@/lib/server-auth"

export async function POST(
  request: NextRequest,
  { params }: { params: Promise<{ userId: string }> }
) {
  const { userId } = await params
  const token = getAccessTokenFromRequest(request)
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
    ...backendAuthHeaders(token),
  }

  let res: Response
  try {
    res = await fetch(
      `${BACKEND_SERVICE_URL}/api/generate/user/${userId}`,
      { method: "POST", headers, body: await request.text() }
    )
  } catch (err) {
    const message = err instanceof Error ? err.message : "Backend unreachable"
    console.error("[/api/generate] connection error:", message)
    return NextResponse.json({ detail: `Backend unreachable: ${message}` }, { status: 502 })
  }

  const text = await res.text()
  const parsed = (() => { try { return JSON.parse(text) } catch { return { detail: text || `Backend error ${res.status}` } } })()
  const nextRes = NextResponse.json(parsed, { status: res.status })
  if (res.status === 401) clearAuthCookieOnResponse(nextRes)
  return nextRes
}
