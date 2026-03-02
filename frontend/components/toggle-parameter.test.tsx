import { render, screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { describe, expect, it, vi } from "vitest"
import { ToggleParameter } from "@/components/toggle-parameter"

describe("ToggleParameter", () => {
  it("renders label, description, and checked state", () => {
    render(
      <ToggleParameter
        label="Mean Reversion"
        description="Enable mean reversion behavior"
        checked={true}
        onChange={vi.fn()}
      />
    )

    expect(screen.getByText("Mean Reversion")).toBeInTheDocument()
    expect(screen.getByText("Enable mean reversion behavior")).toBeInTheDocument()
    expect(screen.getByRole("switch")).toHaveAttribute("aria-checked", "true")
  })

  it("calls onChange when toggled", async () => {
    const user = userEvent.setup()
    const onChange = vi.fn()
    render(
      <ToggleParameter
        label="Fat Tails"
        description="Allow heavier tails"
        checked={false}
        onChange={onChange}
      />
    )

    await user.click(screen.getByRole("switch"))
    expect(onChange).toHaveBeenCalledWith(true)
    expect(onChange).toHaveBeenCalledTimes(1)
  })
})
