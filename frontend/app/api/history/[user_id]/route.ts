import { NextRequest, NextResponse } from "next/server"
import { BACKEND_SERVICE_URL } from "@/lib/server-constants"

export async function GET(
    request: NextRequest,
    { params }: { params: Promise<{ user_id: string }> }
) {
    const { user_id } = await params
    const { searchParams } = new URL(request.url)
    const queryString = searchParams.toString()

    const url = `${BACKEND_SERVICE_URL}/api/history/user/${user_id}${queryString ? `?${queryString}` : ""}`

    try {
        const response = await fetch(url, { method: "GET" })
        const data = await response.json()
        return NextResponse.json(data, { status: response.status })
    } catch (error) {
        console.error("Error fetching job history:", error)
        return NextResponse.json(
            { error: "Failed to fetch job history" },
            { status: 500 }
        )
    }
}
