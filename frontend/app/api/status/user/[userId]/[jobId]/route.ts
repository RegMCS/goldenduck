import { NextRequest, NextResponse } from "next/server"
import { BACKEND_SERVICE_URL } from "@/lib/server-constants"

export async function GET(
  _request: NextRequest,
  { params }: { params: Promise<{ userId: string; jobId: string }> }
) {
  const { userId, jobId } = await params
  const res = await fetch(
    `${BACKEND_SERVICE_URL}/api/status/user/${userId}/${jobId}`
  )
  const data = await res.json()
  return NextResponse.json(data, { status: res.status })
}
