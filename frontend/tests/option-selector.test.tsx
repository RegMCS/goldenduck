import { render, screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { describe, expect, it, vi } from "vitest"
import { OptionSelector } from "@/components/option-selector"

describe("OptionSelector", () => {
  const options = [
    { value: "low", label: "Low", description: "Lower risk" },
    { value: "high", label: "High", description: "Higher risk" },
  ] as const

  it("renders options and highlights the selected value", () => {
    render(<OptionSelector options={[...options]} value="low" onChange={vi.fn()} />)

    expect(screen.getByRole("button", { name: /low/i })).toHaveClass("border-primary")
    expect(screen.getByRole("button", { name: /high/i })).toHaveClass("border-border")
  })

  it("calls onChange with clicked option value", async () => {
    const user = userEvent.setup()
    const onChange = vi.fn()
    render(<OptionSelector options={[...options]} value="low" onChange={onChange} />)

    await user.click(screen.getByRole("button", { name: /high/i }))
    expect(onChange).toHaveBeenCalledWith("high")
    expect(onChange).toHaveBeenCalledTimes(1)
  })
})
