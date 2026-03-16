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
    `${BACKEND_SERVICE_URL}/api/status/user/${userId}/${jobId}`,
    { headers }
  )
  const data = await res.json()
  return NextResponse.json(data, { status: res.status })
}
