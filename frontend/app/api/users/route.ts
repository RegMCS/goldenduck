import { NextRequest, NextResponse } from "next/server"
import { BACKEND_SERVICE_URL } from "@/lib/server-constants"
import { getAccessTokenFromRequest, backendAuthHeaders } from "@/lib/server-auth"

export async function GET(request: NextRequest) {
    const { searchParams } = new URL(request.url)
    const queryString = searchParams.toString()
    const token = getAccessTokenFromRequest(request)
    const headers = backendAuthHeaders(token)

    const url = `${BACKEND_SERVICE_URL}/api/users${queryString ? `?${queryString}` : ""}`

    try {
        const response = await fetch(url, { method: "GET", headers })
        const data = await response.json()
        return NextResponse.json(data, { status: response.status })
    } catch (error) {
        console.error("Error fetching users:", error)
        return NextResponse.json(
            { error: "Failed to fetch users" },
            { status: 500 }
        )
    }
}
