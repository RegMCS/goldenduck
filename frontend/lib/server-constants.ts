// In production (Docker) the backend is reachable via the service name.
// In local dev (npm run dev) it is reachable via localhost.
export const BACKEND_SERVICE_URL =
    process.env.BACKEND_SERVICE_URL ??
    (process.env.NODE_ENV === "development" ? "http://localhost:8000" : "http://job-scheduler:8000")
