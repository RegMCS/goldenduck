import { NextRequest, NextResponse } from "next/server"
import { BACKEND_SERVICE_URL } from "@/lib/server-constants"
import { getAccessTokenFromRequest, backendAuthHeaders } from "@/lib/server-auth"

/**
 * Multipart upload proxy: forwards FormData to the backend without forcing JSON.
 * The catch-all /api/training/[...path] route only supports JSON bodies.
 */
export async function POST(request: NextRequest) {
  const token = getAccessTokenFromRequest(request)
  const formData = await request.formData()
  const url = `${BACKEND_SERVICE_URL}/api/training/models/upload`

  try {
    const response = await fetch(url, {
      method: "POST",
      headers: backendAuthHeaders(token),
      body: formData,
    })
    const contentType = response.headers.get("content-type") ?? ""
    if (contentType.includes("application/json")) {
      const data = await response.json()
      return NextResponse.json(data, { status: response.status })
    }
    const text = await response.text()
    return new NextResponse(text, { status: response.status })
  } catch (error) {
    console.error("Error proxying model upload:", error)
    return NextResponse.json({ error: "Failed to reach training service" }, { status: 502 })
  }
}
