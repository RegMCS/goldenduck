import { render, screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { describe, expect, it, vi } from "vitest"
import { ConfigSummary } from "@/components/config-summary"

function createParameters() {
  return {
    assetClass: "equities",
    trendType: "bullish",
    marketRegime: "high-volatility",
    volatilityLevel: "medium",
    generationModel: "garch",
    timeHorizon: 120,
    meanReversion: true,
    fatTails: false,
    jumpDiffusion: true,
  } as unknown
}

describe("ConfigSummary", () => {
  it("shows continue flow before final step and triggers navigation callbacks", async () => {
    const user = userEvent.setup()
    const onNext = vi.fn()
    const onBack = vi.fn()
    const onReset = vi.fn()

    render(
      <ConfigSummary
        parameters={createParameters()}
        onGenerate={vi.fn()}
        onReset={onReset}
        isGenerating={false}
        currentStep={2}
        totalSteps={3}
        onNext={onNext}
        onBack={onBack}
        canProceed={true}
      />
    )

    await user.click(screen.getByRole("button", { name: /continue/i }))
    await user.click(screen.getByRole("button", { name: /back/i }))
    await user.click(screen.getByRole("button", { name: /reset/i }))

    expect(onNext).toHaveBeenCalledTimes(1)
    expect(onBack).toHaveBeenCalledTimes(1)
    expect(onReset).toHaveBeenCalledTimes(1)
    expect(screen.queryByRole("button", { name: /generate data/i })).not.toBeInTheDocument()
  })

  it("shows generate flow on last step and disables generate while running", () => {
    render(
      <ConfigSummary
        parameters={createParameters()}
        onGenerate={vi.fn()}
        onReset={vi.fn()}
        isGenerating={true}
        currentStep={3}
        totalSteps={3}
        onNext={vi.fn()}
        onBack={vi.fn()}
        canProceed={true}
      />
    )

    expect(screen.getByText("Step 3 of 3")).toBeInTheDocument()
    expect(screen.getByRole("button", { name: /generating/i })).toBeDisabled()
    expect(screen.getByRole("button", { name: /export config/i })).toBeDisabled()
  })
})
