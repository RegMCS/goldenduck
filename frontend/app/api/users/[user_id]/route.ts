import { NextRequest, NextResponse } from "next/server"
import { BACKEND_SERVICE_URL } from "@/lib/server-constants"
import { getAccessTokenFromRequest, backendAuthHeaders } from "@/lib/server-auth"

export async function PUT(
    request: NextRequest,
    { params }: { params: Promise<{ user_id: string }> }
) {
    const { user_id } = await params
    const token = getAccessTokenFromRequest(request)
    const headers = { 
        ...backendAuthHeaders(token),
        "Content-Type": "application/json"
    }

    const url = `${BACKEND_SERVICE_URL}/api/users/${user_id}`

    try {
        const body = await request.text()
        const response = await fetch(url, { 
            method: "PUT", 
            headers,
            body
        })
        const data = await response.json()
        return NextResponse.json(data, { status: response.status })
    } catch (error) {
        console.error("Error updating user admin status:", error)
        return NextResponse.json(
            { error: "Failed to update user" },
            { status: 500 }
        )
    }
}
