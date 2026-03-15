import { render, screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { describe, expect, it, vi } from "vitest"
import { Header } from "@/components/header"

// Mock next-themes
vi.mock("next-themes", () => ({
  useTheme: () => ({
    theme: "light",
    setTheme: vi.fn(),
  }),
}))

// Mock next/navigation (usePathname)
vi.mock("next/navigation", () => ({
  usePathname: () => "/",
}))

describe("Header", () => {
  it("renders the logo and app name", () => {
    render(<Header />)

    expect(screen.getByText("SynthMarket")).toBeInTheDocument()
    expect(screen.getByText("Synthetic Data Generator")).toBeInTheDocument()
    expect(screen.getByRole("banner")).toBeInTheDocument()
  })

  it("renders nav links and action buttons", () => {
    render(<Header />)

    expect(screen.getByRole("link", { name: /configure/i })).toBeInTheDocument()
    expect(screen.getByRole("link", { name: /reports/i })).toBeInTheDocument()
    expect(screen.getByRole("button", { name: /toggle theme/i })).toBeInTheDocument()
    expect(screen.getByRole("button", { name: /settings/i })).toBeInTheDocument()
  })

  it("reports nav link points to /reports", () => {
    render(<Header />)

    const reportsLink = screen.getByRole("link", { name: /reports/i })
    expect(reportsLink).toHaveAttribute("href", "/reports")
    expect(reportsLink).toHaveAttribute("id", "nav-reports-link")
  })

  it("configure nav link points to /", () => {
    render(<Header />)

    const configureLink = screen.getByRole("link", { name: /configure/i })
    expect(configureLink).toHaveAttribute("href", "/")
  })

  it("has proper accessibility attributes", () => {
    render(<Header />)

    expect(screen.getByRole("banner")).toBeInTheDocument()
    expect(screen.getByRole("navigation")).toBeInTheDocument()
    expect(screen.getByRole("button", { name: /toggle theme/i })).toBeInTheDocument()
    expect(screen.getByRole("button", { name: /settings/i })).toBeInTheDocument()
  })

  it("applies sticky positioning", () => {
    render(<Header />)

    const header = screen.getByRole("banner")
    expect(header).toHaveClass("sticky")
    expect(header).toHaveClass("top-0")
    expect(header).toHaveClass("z-50")
  })
})