import { NextRequest, NextResponse } from "next/server"
import { getAccessTokenCookieName } from "@/lib/server-auth"

// Pages that don't require authentication
const PUBLIC_PATHS = ["/login", "/register"]

// Prefix patterns that are always public (API routes, static assets, etc.)
const PUBLIC_PREFIXES = ["/api/", "/_next/", "/favicon"]

export function middleware(request: NextRequest) {
  const { pathname } = request.nextUrl

  // Always allow public prefixes
  if (PUBLIC_PREFIXES.some((p) => pathname.startsWith(p))) {
    return NextResponse.next()
  }

  // Always allow public pages
  if (PUBLIC_PATHS.includes(pathname)) {
    return NextResponse.next()
  }

  const token = request.cookies.get(getAccessTokenCookieName())?.value

  // Dev mode: no token required
  if (!token && process.env.NODE_ENV === "development") {
    return NextResponse.next()
  }

  // Not authenticated → redirect to login, preserving the intended destination
  if (!token) {
    const loginUrl = request.nextUrl.clone()
    loginUrl.pathname = "/login"
    loginUrl.searchParams.set("next", pathname)
    return NextResponse.redirect(loginUrl)
  }

  return NextResponse.next()
}

export const config = {
  // Run on all routes except Next.js internals and static files
  matcher: ["/((?!_next/static|_next/image|favicon.ico).*)"],
}
