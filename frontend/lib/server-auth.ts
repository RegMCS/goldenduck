import { NextRequest, NextResponse } from "next/server"

const ACCESS_TOKEN_COOKIE = "access_token"

/**
 * Read the JWT from the request cookie (set by login).
 * Use this in API routes that proxy to the backend so we can send Authorization: Bearer.
 */
export function getAccessTokenFromRequest(request: NextRequest): string | null {
  const value = request.cookies.get(ACCESS_TOKEN_COOKIE)?.value ?? null
  return value ? value.trim() : null
}

/** Headers to forward the JWT to the backend. Token is assumed already trimmed (e.g. from getAccessTokenFromRequest). */
export function backendAuthHeaders(token: string | null): Record<string, string> {
  const headers: Record<string, string> = {}
  if (token) headers["Authorization"] = `Bearer ${token}`
  return headers
}

export function getAccessTokenCookieName(): string {
  return ACCESS_TOKEN_COOKIE
}

/** Clear the auth cookie on a response (logout or invalid session). Options must match login so the browser removes it. */
export function clearAuthCookieOnResponse(res: NextResponse): void {
  res.cookies.set(ACCESS_TOKEN_COOKIE, "", {
    httpOnly: true,
    secure: process.env.NODE_ENV === "production",
    sameSite: "lax",
    path: "/",
    maxAge: 0,
    expires: new Date(0),
  })
}
