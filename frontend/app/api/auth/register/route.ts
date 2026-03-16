import { NextRequest, NextResponse } from "next/server"
import { BACKEND_SERVICE_URL } from "@/lib/server-constants"

export async function POST(request: NextRequest) {
    try {
        const body = await request.json()
        const url = `${BACKEND_SERVICE_URL}/api/auth/register`

        const response = await fetch(url, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(body)
        })

        const data = await response.json()
        return NextResponse.json(data, { status: response.status })
    } catch (error) {
        console.error("Error calling backend register API:", error)
        return NextResponse.json(
            { error: "Failed to register" },
            { status: 500 }
        )
    }
}
