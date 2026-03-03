import { NextRequest, NextResponse } from "next/server"

const BACKEND_URL = process.env.API_GATEWAY_URL || "http://job-scheduler:8000"

export async function GET(
  _request: NextRequest,
  { params }: { params: Promise<{ userId: string; jobId: string }> }
) {
  const { userId, jobId } = await params
  const res = await fetch(
    `${BACKEND_URL}/api/status/user/${userId}/${jobId}`
  )
  const data = await res.json()
  return NextResponse.json(data, { status: res.status })
}
