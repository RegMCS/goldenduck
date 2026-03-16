import { NextRequest, NextResponse } from "next/server"

const ACCESS_TOKEN_COOKIE = "access_token"

const PUBLIC_PATHS = new Set(["/login", "/register"])
const PUBLIC_PREFIXES = ["/api/", "/_next/", "/favicon"]

export function middleware(request: NextRequest) {
  const { pathname } = request.nextUrl

  if (PUBLIC_PREFIXES.some((p) => pathname.startsWith(p))) {
    return NextResponse.next()
  }

  if (PUBLIC_PATHS.has(pathname)) {
    return NextResponse.next()
  }

  const token = request.cookies.get(ACCESS_TOKEN_COOKIE)?.value

  if (!token && process.env.NODE_ENV === "development") {
    return NextResponse.next()
  }

  if (!token) {
    const loginUrl = request.nextUrl.clone()
    loginUrl.pathname = "/login"
    loginUrl.searchParams.set("next", pathname)
    return NextResponse.redirect(loginUrl)
  }

  return NextResponse.next()
}

export const config = {
  matcher: ["/((?!_next/static|_next/image|favicon.ico).*)"],
}
