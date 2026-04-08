/**
 * Deduplicates identical in-flight GET requests by URL.
 * Next.js dev runs React Strict Mode, which mounts effects twice in a row; without this,
 * the same tab would issue duplicate fetches for uploads, runs, evaluation, etc.
 */
const inFlight = new Map<string, Promise<Response>>()

export function fetchTrainingGet(url: string, init?: RequestInit): Promise<Response> {
  const method = (init?.method ?? "GET").toUpperCase()
  if (method !== "GET") {
    return fetch(url, init)
  }
  const key = `${method} ${url}`
  const existing = inFlight.get(key)
  if (existing) {
    return existing.then((response) => response.clone())
  }
  const pending = fetch(url, { ...init, method: "GET" })
  inFlight.set(key, pending)
  pending.finally(() => {
    inFlight.delete(key)
  })
  return pending.then((response) => response.clone())
}
