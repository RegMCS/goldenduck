import { NextRequest, NextResponse } from "next/server"
import { BACKEND_SERVICE_URL } from "@/lib/server-constants"
import { getAccessTokenFromRequest, backendAuthHeaders } from "@/lib/server-auth"

/**
 * Catch-all proxy for /api/training/* → backend /api/training/*
 * Forwards the JWT cookie as Authorization: Bearer so the backend
 * can enforce admin-only access on training routes.
 */
async function proxyRequest(
  request: NextRequest,
  { params }: { params: Promise<{ path: string[] }> },
  method: string,
) {
  const resolvedParams = await params
  const subPath = resolvedParams.path.join("/")
  const { searchParams } = new URL(request.url)
  const queryString = searchParams.toString()
  const url = `${BACKEND_SERVICE_URL}/api/training/${subPath}${queryString ? `?${queryString}` : ""}`

  const token = getAccessTokenFromRequest(request)
  const baseHeaders = backendAuthHeaders(token)

  const options: RequestInit = { method, headers: baseHeaders }

  if (method === "POST" || method === "PUT" || method === "PATCH") {
    options.headers = { ...baseHeaders, "Content-Type": "application/json" }
    options.body = await request.text()
  }

  try {
    const response = await fetch(url, options)
    const contentType = response.headers.get("content-type") ?? ""
    if (contentType.includes("application/json")) {
      const data = await response.json()
      return NextResponse.json(data, { status: response.status })
    }
    const text = await response.text()
    return new NextResponse(text, { status: response.status })
  } catch (error) {
    console.error("Error proxying to training backend:", error)
    return NextResponse.json({ error: "Failed to reach training service" }, { status: 502 })
  }
}

export async function GET(request: NextRequest, ctx: { params: Promise<{ path: string[] }> }) {
  return proxyRequest(request, ctx, "GET")
}

export async function POST(request: NextRequest, ctx: { params: Promise<{ path: string[] }> }) {
  return proxyRequest(request, ctx, "POST")
}

export async function DELETE(request: NextRequest, ctx: { params: Promise<{ path: string[] }> }) {
  return proxyRequest(request, ctx, "DELETE")
}
