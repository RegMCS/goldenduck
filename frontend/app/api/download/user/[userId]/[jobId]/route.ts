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
    `${BACKEND_SERVICE_URL}/api/download/user/${userId}/${jobId}`,
    { redirect: "manual", headers }
  )

  // Backend returns a 307 redirect to a presigned S3 URL — forward it to the browser
  const location = res.headers.get("location")
  if (location) {
    return NextResponse.redirect(location)
  }

  const data = await res.json().catch(() => ({ detail: "Download unavailable" }))
  return NextResponse.json(data, { status: res.status })
}
