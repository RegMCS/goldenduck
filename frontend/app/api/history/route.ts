import { NextRequest, NextResponse } from "next/server"
import { BACKEND_SERVICE_URL } from "@/lib/server-constants"
import { getAccessTokenFromRequest, backendAuthHeaders } from "@/lib/server-auth"

async function proxyRequest(request: NextRequest, method: string) {
  const { searchParams } = new URL(request.url)
  const queryString = searchParams.toString()
  const url = `${BACKEND_SERVICE_URL}/api/history${queryString ? `?${queryString}` : ""}`
  const token = getAccessTokenFromRequest(request)
  const headers = backendAuthHeaders(token)

  const options: RequestInit = { method, headers }

  if (method === "POST") {
    options.headers = { ...headers, "Content-Type": "application/json" }
    options.body = await request.text()
  }

  const response = await fetch(url, options)
  const data = await response.json()

  return NextResponse.json(data, { status: response.status })
}

export async function GET(request: NextRequest) {
  return proxyRequest(request, "GET")
}

export async function POST(request: NextRequest) {
  return proxyRequest(request, "POST")
}

export async function DELETE(request: NextRequest) {
  return proxyRequest(request, "DELETE")
}
