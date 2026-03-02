import { render, screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { describe, expect, it, vi } from "vitest"
import { StepIndicator } from "@/components/step-indicator"

describe("StepIndicator", () => {
  const steps = [
    { id: 1, label: "Upload" },
    { id: 2, label: "Configure" },
    { id: 3, label: "Generate" },
  ]

  it("disables future steps and allows previous/current steps", async () => {
    const onStepClick = vi.fn()
    const user = userEvent.setup()
    render(<StepIndicator steps={steps} currentStep={2} onStepClick={onStepClick} />)

    const uploadBtn = screen.getByRole("button", { name: /upload/i })
    const configureBtn = screen.getByRole("button", { name: /configure/i })
    const generateBtn = screen.getByRole("button", { name: /generate/i })

    expect(uploadBtn).toBeEnabled()
    expect(configureBtn).toBeEnabled()
    expect(generateBtn).toBeDisabled()

    await user.click(uploadBtn)
    await user.click(configureBtn)
    await user.click(generateBtn)

    expect(onStepClick).toHaveBeenCalledWith(1)
    expect(onStepClick).toHaveBeenCalledWith(2)
    expect(onStepClick).toHaveBeenCalledTimes(2)
  })
})
