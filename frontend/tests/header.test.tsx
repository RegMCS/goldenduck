import { render, screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { describe, expect, it, vi } from "vitest"
import { Header } from "@/components/header"
import { AuthProvider } from "@/components/auth-provider"

// Mock next-themes
vi.mock("next-themes", () => ({
  useTheme: () => ({
    theme: "light",
    setTheme: vi.fn(),
  }),
}))

// Mock next/navigation
vi.mock("next/navigation", () => ({
  usePathname: () => "/",
  useRouter: () => ({ push: vi.fn(), replace: vi.fn(), refresh: vi.fn() }),
}))

// Mock fetch so AuthProvider's /api/auth/me call doesn't fail
global.fetch = vi.fn(() =>
  Promise.resolve({ ok: false, json: () => Promise.resolve({}) } as Response)
)

const renderWithAuth = (ui: React.ReactElement) =>
  render(<AuthProvider>{ui}</AuthProvider>)

describe("Header", () => {
  it("renders the logo and app name", () => {
    renderWithAuth(<Header />)

    expect(screen.getByText("SynthMarket")).toBeInTheDocument()
    expect(screen.getByText("Synthetic Data Generator")).toBeInTheDocument()
    expect(screen.getByRole("banner")).toBeInTheDocument()
  })

  it("renders nav links and action buttons", () => {
    renderWithAuth(<Header />)

    expect(screen.getByRole("link", { name: /configure/i })).toBeInTheDocument()
    expect(screen.getByRole("link", { name: /reports/i })).toBeInTheDocument()
    expect(screen.getByRole("button", { name: /toggle theme/i })).toBeInTheDocument()
  })

  it("reports nav link points to /reports", () => {
    renderWithAuth(<Header />)

    const reportsLink = screen.getByRole("link", { name: /reports/i })
    expect(reportsLink).toHaveAttribute("href", "/reports")
    expect(reportsLink).toHaveAttribute("id", "nav-reports-link")
  })

  it("configure nav link points to /", () => {
    renderWithAuth(<Header />)

    const configureLink = screen.getByRole("link", { name: /configure/i })
    expect(configureLink).toHaveAttribute("href", "/")
  })

  it("has proper accessibility attributes", () => {
    renderWithAuth(<Header />)

    expect(screen.getByRole("banner")).toBeInTheDocument()
    expect(screen.getByRole("navigation")).toBeInTheDocument()
    expect(screen.getByRole("button", { name: /toggle theme/i })).toBeInTheDocument()
  })

  it("applies sticky positioning", () => {
    renderWithAuth(<Header />)

    const header = screen.getByRole("banner")
    expect(header).toHaveClass("sticky")
    expect(header).toHaveClass("top-0")
    expect(header).toHaveClass("z-50")
  })
})