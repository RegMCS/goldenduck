import { NextRequest, NextResponse } from "next/server"
import { BACKEND_SERVICE_URL } from "@/lib/server-constants"
import { getAccessTokenFromRequest, backendAuthHeaders } from "@/lib/server-auth"

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ userId: string; jobId: string }> }
) {
  const { userId, jobId } = await params
  const token = getAccessTokenFromRequest(request)
  const headers = backendAuthHeaders(token)

  const res = await fetch(
    `${BACKEND_SERVICE_URL}/api/download/user/${userId}/${jobId}/selected-path`,
    { redirect: "manual", headers }
  )

  // Keep compatibility with redirect-based downloads if backend ever returns one.
  const location = res.headers.get("location")
  if (location) {
    return NextResponse.redirect(location)
  }

  if (!res.ok) {
    const data = await res.json().catch(() => ({ detail: "Download unavailable" }))
    return NextResponse.json(data, { status: res.status })
  }

  const body = await res.arrayBuffer()
  const contentType = res.headers.get("content-type") ?? "text/csv"
  const contentDisposition =
    res.headers.get("content-disposition") ??
    `attachment; filename="${jobId}_selected_path_ohlcv.csv"`

  return new NextResponse(body, {
    status: 200,
    headers: {
      "Content-Type": contentType,
      "Content-Disposition": contentDisposition,
    },
  })
}
