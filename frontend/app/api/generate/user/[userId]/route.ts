import { NextRequest, NextResponse } from "next/server"
import { BACKEND_SERVICE_URL } from "@/lib/server-constants"

export async function POST(
  request: NextRequest,
  { params }: { params: Promise<{ userId: string }> }
) {
  const { userId } = await params
  let res: Response
  try {
    res = await fetch(
      `${BACKEND_SERVICE_URL}/api/generate/user/${userId}`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: await request.text(),
      }
    )
  } catch (err) {
    const message = err instanceof Error ? err.message : "Backend unreachable"
    console.error("[/api/generate] connection error:", message)
    return NextResponse.json({ detail: `Backend unreachable: ${message}` }, { status: 502 })
  }

  const text = await res.text()
  console.error(`[/api/generate] backend responded ${res.status}:`, text)
  try {
    return NextResponse.json(JSON.parse(text), { status: res.status })
  } catch {
    return NextResponse.json({ detail: text || `Backend error ${res.status}` }, { status: res.status })
  }
}
