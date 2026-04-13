import { NextRequest, NextResponse } from "next/server"
import { BACKEND_SERVICE_URL } from "@/lib/server-constants"

export async function GET(request: NextRequest) {
  try {
    const res = await fetch(`${BACKEND_SERVICE_URL}/api/default-csv`)

    if (!res.ok) {
      return NextResponse.json(
        { detail: `Backend error: ${res.statusText}` },
        { status: res.status }
      )
    }

    const contentType = res.headers.get("content-type") ?? ""

    // If it's a JSON response (presigned URL for S3), pass it through
    if (contentType.includes("application/json")) {
      const data = await res.json()
      return NextResponse.json(data, { status: 200 })
    }

    // If it's a CSV file (development), stream it directly
    const blob = await res.blob()
    return new NextResponse(blob, {
      status: 200,
      headers: {
        "Content-Type": "text/csv",
        "Content-Disposition": 'attachment; filename="AAPL_real.csv"',
      },
    })
  } catch (err) {
    const message = err instanceof Error ? err.message : "Backend unreachable"
    return NextResponse.json(
      { detail: `Backend unreachable: ${message}` },
      { status: 502 }
    )
  }
}
