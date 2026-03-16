import { NextResponse } from "next/server"
import { clearAuthCookieOnResponse } from "@/lib/server-auth"

export async function POST() {
  const res = NextResponse.json({ success: true })
  clearAuthCookieOnResponse(res)
  return res
}
