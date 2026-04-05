import { NextRequest, NextResponse } from "next/server"
import { BACKEND_SERVICE_URL } from "@/lib/server-constants"
import { getAccessTokenCookieName } from "@/lib/server-auth"

export async function POST(request: NextRequest) {
    try {
        const body = await request.json()
        const url = `${BACKEND_SERVICE_URL}/api/auth/login`

        const params = new URLSearchParams()
        params.append('username', body.username)
        params.append('password', body.password)

        const response = await fetch(url, {
            method: "POST",
            headers: { "Content-Type": "application/x-www-form-urlencoded" },
            body: params.toString()
        })

        const data = await response.json()
        if (!response.ok) {
            return NextResponse.json(data, { status: response.status })
        }

        const token = (data.access_token ?? "").trim()
        const user = data.user
            ? {
                  id: String(data.user.id),
                  username: data.user.username,
                  first_name: data.user.first_name ?? null,
                  last_name: data.user.last_name ?? null,
                  is_admin: Boolean(data.user.is_admin),
              }
            : null

        const res = NextResponse.json({ success: true, user })
        res.cookies.set(getAccessTokenCookieName(), token, {
            httpOnly: true,
            secure: process.env.NODE_ENV === "production",
            sameSite: "lax",
            path: "/",
            maxAge: 7 * 24 * 60 * 60, // 7 days
        })

        return res
    } catch (error) {
        console.error("Error calling backend login API:", error)
        return NextResponse.json(
            { error: "Failed to login" },
            { status: 500 }
        )
    }
}
