"use client"

import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Progress } from "@/components/ui/progress"
import {
  Download,
  RotateCcw,
  ChevronRight,
  ChevronLeft,
  Sparkles,
} from "lucide-react"
import type { PresetMarketParameters } from "@/lib/types"

interface ConfigSummaryProps {
  parameters: PresetMarketParameters
  onGenerate: () => void
  onReset: () => void
  isGenerating?: boolean
  currentStep: number
  totalSteps: number
  onNext: () => void
  onBack: () => void
  canProceed: boolean
}

export function ConfigSummary({
  parameters,
  onGenerate,
  onReset,
  isGenerating,
  currentStep,
  totalSteps,
  onNext,
  onBack,
  canProceed,
}: ConfigSummaryProps) {
  const progress = (currentStep / totalSteps) * 100
  const isLastStep = currentStep === totalSteps

  const summaryItems = [
    {
      label: "Asset Class",
      value: parameters.assetClass.toUpperCase(),
      filled: true,
    },
    {
      label: "Trend",
      value: parameters.trendType,
      filled: true,
    },
    {
      label: "Regime",
      value: parameters.marketRegime.replace("-", " "),
      filled: currentStep >= 2,
    },
    {
      label: "Volatility",
      value: parameters.volatilityLevel,
      filled: currentStep >= 2,
    },
    {
      label: "Model",
      value: parameters.generationModel.toUpperCase(),
      filled: currentStep >= 3,
    },
    {
      label: "Time Horizon",
      value: `${parameters.timeHorizon} days`,
      filled: currentStep >= 3,
    },
  ]

  return (
    <Card className="sticky top-6 border-border bg-card overflow-hidden">
      <CardHeader className="pb-3 border-b border-border">
        <div className="flex items-center justify-between">
          <CardTitle className="text-base font-semibold text-foreground">
            Configuration
          </CardTitle>
          <span className="text-xs text-muted-foreground">
            Step {currentStep} of {totalSteps}
          </span>
        </div>
        <Progress value={progress} className="h-1.5 mt-2" />
      </CardHeader>

      <CardContent className="pt-4 space-y-4">
        <div className="space-y-2">
          {summaryItems.map((item) => (
            <div
              key={item.label}
              className={`flex justify-between text-sm transition-opacity ${
                item.filled ? "opacity-100" : "opacity-40"
              }`}
            >
              <span className="text-muted-foreground">{item.label}</span>
              <span className="font-medium capitalize text-foreground">
                {item.filled ? item.value : "---"}
              </span>
            </div>
          ))}
        </div>

        <div className="border-t border-border pt-3">
          <p className="text-xs font-medium text-muted-foreground mb-2">
            Advanced Options
          </p>
          <div className="flex flex-wrap gap-1.5">
            {[
              { label: "Mean Reversion", active: parameters.meanReversion },
              { label: "Fat Tails", active: parameters.fatTails },
              { label: "Jump Diffusion", active: parameters.jumpDiffusion },
            ].map((opt) => (
              <span
                key={opt.label}
                className={`inline-flex items-center rounded-full px-2 py-0.5 text-xs transition-all ${
                  opt.active
                    ? "bg-primary/20 text-primary"
                    : "bg-secondary text-muted-foreground"
                }`}
              >
                {opt.label}
              </span>
            ))}
          </div>
        </div>

        <div className="flex flex-col gap-2 pt-2 border-t border-border">
          {isLastStep ? (
            <Button
              onClick={onGenerate}
              disabled={isGenerating}
              className="w-full gap-2"
              size="lg"
            >
              {isGenerating ? (
                <>
                  <div className="h-4 w-4 animate-spin rounded-full border-2 border-primary-foreground border-t-transparent" />
                  Generating...
                </>
              ) : (
                <>
                  <Sparkles className="h-4 w-4" />
                  Generate Data
                </>
              )}
            </Button>
          ) : (
            <Button
              onClick={onNext}
              disabled={!canProceed}
              className="w-full gap-2"
              size="lg"
            >
              Continue
              <ChevronRight className="h-4 w-4" />
            </Button>
          )}

          <div className="flex gap-2">
            <Button
              variant="outline"
              onClick={onBack}
              disabled={currentStep === 1}
              className="flex-1 gap-2 bg-transparent"
              size="sm"
            >
              <ChevronLeft className="h-4 w-4" />
              Back
            </Button>
            <Button
              variant="outline"
              onClick={onReset}
              className="flex-1 gap-2 bg-transparent"
              size="sm"
            >
              <RotateCcw className="h-4 w-4" />
              Reset
            </Button>
          </div>

          {isLastStep && (
            <Button
              variant="ghost"
              disabled
              className="w-full gap-2 text-muted-foreground"
              size="sm"
            >
              <Download className="h-4 w-4" />
              Export Config
            </Button>
          )}
        </div>
      </CardContent>
    </Card>
  )
}
