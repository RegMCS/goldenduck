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

describe("Header", () => {
  it("renders the logo and app name", () => {
    render(<Header />)
    
    expect(screen.getByText("SynthMarket")).toBeInTheDocument()
    expect(screen.getByText("Synthetic Data Generator")).toBeInTheDocument()
    expect(screen.getByRole("banner")).toBeInTheDocument()
  })

  it("renders navigation buttons", () => {
    render(<Header />)
    
    expect(screen.getByRole("button", { name: /help/i })).toBeInTheDocument()
    expect(screen.getByRole("button", { name: /toggle theme/i })).toBeInTheDocument()
    expect(screen.getByRole("button", { name: /settings/i })).toBeInTheDocument()
  })

  it("shows help dropdown menu when clicked", async () => {
    const user = userEvent.setup()
    render(<Header />)
    
    const helpButton = screen.getByRole("button", { name: /help/i })
    await user.click(helpButton)
    
    expect(screen.getByText("Getting Started")).toBeInTheDocument()
    expect(screen.getByText("Documentation")).toBeInTheDocument()
    expect(screen.getByText("Model Guide")).toBeInTheDocument()
    expect(screen.getByText("Contact Support")).toBeInTheDocument()
  })

  it("has proper accessibility attributes", () => {
    render(<Header />)
    
    // Check that the header exists and has basic accessibility
    expect(screen.getByRole("banner")).toBeInTheDocument()
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