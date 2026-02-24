import { NextRequest, NextResponse } from "next/server"

const BACKEND_URL = process.env.API_GATEWAY_URL || "http://job-scheduler:8000"

export async function GET(
  _request: NextRequest,
  { params }: { params: Promise<{ userId: string; jobId: string }> }
) {
  const { userId, jobId } = await params
  const res = await fetch(
    `${BACKEND_URL}/api/download/user/${userId}/${jobId}`,
    { redirect: "manual" }
  )

  // Backend returns a 307 redirect to a presigned S3 URL — forward it to the browser
  const location = res.headers.get("location")
  if (location) {
    return NextResponse.redirect(location)
  }

  const data = await res.json().catch(() => ({ detail: "Download unavailable" }))
  return NextResponse.json(data, { status: res.status })
}
