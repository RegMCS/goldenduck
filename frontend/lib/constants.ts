// ── Development user identity ─────────────────────────────────────────────────
// Fixed UUID used for all requests during development.
// Replace with a real auth session lookup before going to production.
export const DEV_USER_ID = "00000000-0000-0000-0000-000000000001"

export function getUserId(): string {
    return DEV_USER_ID
}

// ── Other shared app constants ────────────────────────────────────────────────
export const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? ""
