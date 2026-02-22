"use client"

import React from "react"

import { useState } from "react"
import {
  BarChart3,
  TrendingUp,
  TrendingDown,
  Minus,
  Activity,
  Gauge,
  Cpu,
  Clock,
  Sliders,
  DollarSign,
  Globe,
  LineChart,
  Gem,
  Bitcoin,
  AlertTriangle,
  Waves,
  ArrowUpRight,
  ArrowDownRight,
  Sparkles,
  Brain,
  Layers,
} from "lucide-react"
import { VisualOptionCard } from "@/components/visual-option-card"
import { SliderParameter } from "@/components/slider-parameter"
import { ToggleParameter } from "@/components/toggle-parameter"
import { ConfigSummary } from "@/components/config-summary"
import { StepIndicator } from "@/components/step-indicator"
import { PresetSelector } from "@/components/preset-selector"
import { InfoTooltip } from "@/components/info-tooltip"
import { type Preset } from "@/lib/presets"
import {
  type MarketParameters,
  defaultParameters,
  type AssetClass,
  type TrendType,
  type MarketRegime,
  type VolatilityLevel,
  type GenerationModel,
} from "@/lib/types"

import { useRouter } from 'next/navigation'

const steps = [
  { id: 1, label: "Market Setup" },
  { id: 2, label: "Conditions" },
  { id: 3, label: "Generation" },
  { id: 4, label: "Fine-Tune" },
]

const assetClassConfig: {
  value: AssetClass
  label: string
  description: string
  icon: React.ReactNode
}[] = [
  {
    value: "equities",
    label: "Equities",
    description: "Stocks & indices",
    icon: <BarChart3 className="h-6 w-6" />,
  },
  {
    value: "fx",
    label: "FX",
    description: "Currency pairs",
    icon: <Globe className="h-6 w-6" />,
  },
  {
    value: "derivatives",
    label: "Derivatives",
    description: "Options & futures",
    icon: <LineChart className="h-6 w-6" />,
  },
  {
    value: "commodities",
    label: "Commodities",
    description: "Metals & energy",
    icon: <Gem className="h-6 w-6" />,
  },
  {
    value: "crypto",
    label: "Crypto",
    description: "Digital assets",
    icon: <Bitcoin className="h-6 w-6" />,
  },
]

const trendTypeConfig: {
  value: TrendType
  label: string
  description: string
  icon: React.ReactNode
  color: string
}[] = [
  {
    value: "bullish",
    label: "Bullish",
    description: "Upward trend",
    icon: <TrendingUp className="h-6 w-6" />,
    color: "#22c55e",
  },
  {
    value: "bearish",
    label: "Bearish",
    description: "Downward trend",
    icon: <TrendingDown className="h-6 w-6" />,
    color: "#ef4444",
  },
  {
    value: "sideways",
    label: "Sideways",
    description: "Range-bound",
    icon: <Minus className="h-6 w-6" />,
    color: "#6b7280",
  },
  {
    value: "volatile",
    label: "Volatile",
    description: "No clear direction",
    icon: <Activity className="h-6 w-6" />,
    color: "#f59e0b",
  },
]

const marketRegimeConfig: {
  value: MarketRegime
  label: string
  description: string
  icon: React.ReactNode
}[] = [
  {
    value: "normal",
    label: "Normal",
    description: "Typical conditions",
    icon: <DollarSign className="h-6 w-6" />,
  },
  {
    value: "crisis",
    label: "Crisis",
    description: "2008-level stress",
    icon: <AlertTriangle className="h-6 w-6" />,
  },
  {
    value: "high-volatility",
    label: "High Vol",
    description: "VIX > 30",
    icon: <Waves className="h-6 w-6" />,
  },
  {
    value: "low-volatility",
    label: "Low Vol",
    description: "Compressed vol",
    icon: <Minus className="h-6 w-6" />,
  },
  {
    value: "recovery",
    label: "Recovery",
    description: "Post-crisis bounce",
    icon: <ArrowUpRight className="h-6 w-6" />,
  },
]

const volatilityLevelConfig: {
  value: VolatilityLevel
  label: string
  description: string
  color: string
}[] = [
  { value: "low", label: "Low", description: "< 10% annual", color: "#22c55e" },
  { value: "medium", label: "Medium", description: "10-20% annual", color: "#3b82f6" },
  { value: "high", label: "High", description: "20-40% annual", color: "#f59e0b" },
  { value: "extreme", label: "Extreme", description: "> 40% annual", color: "#ef4444" },
]

const generationModelConfig: {
  value: GenerationModel
  label: string
  description: string
  icon: React.ReactNode
  tooltip: string
}[] = [
  {
    value: "garch",
    label: "GARCH",
    description: "Traditional volatility model",
    icon: <Gauge className="h-6 w-6" />,
    tooltip:
      "Generalized Autoregressive Conditional Heteroskedasticity - captures volatility clustering common in financial markets",
  },
  {
    value: "gan",
    label: "Conditional GAN",
    description: "Deep generative network",
    icon: <Brain className="h-6 w-6" />,
    tooltip:
      "Generative Adversarial Network trained on real market data to produce realistic synthetic time series",
  },
  {
    value: "ddpm",
    label: "DDPM",
    description: "Diffusion model",
    icon: <Sparkles className="h-6 w-6" />,
    tooltip:
      "Denoising Diffusion Probabilistic Model - state-of-the-art generative model for high-fidelity data synthesis",
  },
]

export function ParameterizationForm() {
  const router = useRouter()
  const [parameters, setParameters] = useState<MarketParameters>(defaultParameters)
  const [isGenerating, setIsGenerating] = useState(false)
  const [currentStep, setCurrentStep] = useState(1)

  const updateParameter = <K extends keyof MarketParameters>(
    key: K,
    value: MarketParameters[K]
  ) => {
    setParameters((prev) => ({ ...prev, [key]: value }))
  }

  const handleGenerate = async () => {
    setIsGenerating(true)
    try{
      // Convert parameters to query string
      const params = new URLSearchParams()
      Object.entries(parameters).forEach(([key, value]) => {
        params.append(key, String(value))
    })

    // Call API to generate visualization
      const response = await fetch(`/api/generate?${params.toString()}`)
      if (!response.ok) throw new Error('Failed to generate data')
      
      // Navigate to results page with parameters
      router.push(`/results?${params.toString()}`)
    } catch (error) {
      console.error('Error generating data:', error)
      setIsGenerating(false)
    }
    // setTimeout(() => {
    //   setIsGenerating(false)
    // }, 2000)
  }

  const handleReset = () => {
    setParameters(defaultParameters)
    setCurrentStep(1)
  }

  const handlePresetSelect = (preset: Preset) => {
    setParameters((prev) => ({ ...prev, ...preset.parameters }))
  }

  const handleNext = () => {
    if (currentStep < steps.length) {
      setCurrentStep(currentStep + 1)
    }
  }

  const handleBack = () => {
    if (currentStep > 1) {
      setCurrentStep(currentStep - 1)
    }
  }

  const renderStepContent = () => {
    switch (currentStep) {
      case 1:
        return (
          <div className="space-y-8">
            <PresetSelector onSelect={handlePresetSelect} currentParams={parameters} />

            <section>
              <div className="mb-4 flex items-center gap-2">
                <h3 className="text-lg font-semibold text-foreground">Select Asset Class</h3>
                <InfoTooltip content="Choose the type of financial instrument you want to simulate. Each asset class has unique statistical properties." />
              </div>
              <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-5">
                {assetClassConfig.map((asset) => (
                  <VisualOptionCard
                    key={asset.value}
                    icon={asset.icon}
                    label={asset.label}
                    description={asset.description}
                    selected={parameters.assetClass === asset.value}
                    onClick={() => updateParameter("assetClass", asset.value)}
                  />
                ))}
              </div>
            </section>

            <section>
              <div className="mb-4 flex items-center gap-2">
                <h3 className="text-lg font-semibold text-foreground">Market Direction</h3>
                <InfoTooltip content="Define the primary directional bias of the simulated market." />
              </div>
              <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
                {trendTypeConfig.map((trend) => (
                  <VisualOptionCard
                    key={trend.value}
                    icon={trend.icon}
                    label={trend.label}
                    description={trend.description}
                    selected={parameters.trendType === trend.value}
                    onClick={() => updateParameter("trendType", trend.value)}
                    accentColor={trend.color}
                  />
                ))}
              </div>
            </section>
          </div>
        )

      case 2:
        return (
          <div className="space-y-8">
            <section>
              <div className="mb-4 flex items-center gap-2">
                <h3 className="text-lg font-semibold text-foreground">Market Regime</h3>
                <InfoTooltip content="Set the overall market environment and stress level for your simulation." />
              </div>
              <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-5">
                {marketRegimeConfig.map((regime) => (
                  <VisualOptionCard
                    key={regime.value}
                    icon={regime.icon}
                    label={regime.label}
                    description={regime.description}
                    selected={parameters.marketRegime === regime.value}
                    onClick={() => updateParameter("marketRegime", regime.value)}
                  />
                ))}
              </div>
            </section>

            <section>
              <div className="mb-4 flex items-center gap-2">
                <h3 className="text-lg font-semibold text-foreground">Volatility Level</h3>
                <InfoTooltip content="Control the magnitude of price fluctuations in annualized terms." />
              </div>
              <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
                {volatilityLevelConfig.map((vol) => (
                  <button
                    key={vol.value}
                    type="button"
                    onClick={() => updateParameter("volatilityLevel", vol.value)}
                    className={`relative flex flex-col items-center gap-1 rounded-xl border-2 p-4 transition-all ${
                      parameters.volatilityLevel === vol.value
                        ? "border-primary bg-primary/10"
                        : "border-transparent bg-secondary/50 hover:bg-secondary hover:border-border"
                    }`}
                  >
                    <div
                      className="h-2 w-full rounded-full mb-2"
                      style={{
                        background: `linear-gradient(to right, ${vol.color}40, ${vol.color})`,
                      }}
                    />
                    <span
                      className={`text-sm font-semibold ${
                        parameters.volatilityLevel === vol.value ? "text-primary" : "text-foreground"
                      }`}
                    >
                      {vol.label}
                    </span>
                    <span className="text-xs text-muted-foreground">{vol.description}</span>
                  </button>
                ))}
              </div>
            </section>
          </div>
        )

      case 3:
        return (
          <div className="space-y-8">
            <section>
              <div className="mb-4 flex items-center gap-2">
                <h3 className="text-lg font-semibold text-foreground">Generation Model</h3>
                <InfoTooltip content="Choose the AI model architecture for generating synthetic data." />
              </div>
              <div className="grid gap-4 lg:grid-cols-3">
                {generationModelConfig.map((model) => (
                  <button
                    key={model.value}
                    type="button"
                    onClick={() => updateParameter("generationModel", model.value)}
                    className={`relative flex flex-col items-start gap-3 rounded-xl border-2 p-5 text-left transition-all ${
                      parameters.generationModel === model.value
                        ? "border-primary bg-primary/10"
                        : "border-transparent bg-secondary/50 hover:bg-secondary hover:border-border"
                    }`}
                  >
                    <div className="flex w-full items-start justify-between">
                      <div
                        className={`flex h-12 w-12 items-center justify-center rounded-xl ${
                          parameters.generationModel === model.value
                            ? "bg-primary/20 text-primary"
                            : "bg-muted text-muted-foreground"
                        }`}
                      >
                        {model.icon}
                      </div>
                      <InfoTooltip content={model.tooltip} />
                    </div>
                    <div>
                      <span
                        className={`text-base font-semibold ${
                          parameters.generationModel === model.value
                            ? "text-primary"
                            : "text-foreground"
                        }`}
                      >
                        {model.label}
                      </span>
                      <p className="mt-1 text-sm text-muted-foreground">{model.description}</p>
                    </div>
                  </button>
                ))}
              </div>
            </section>

            <section className="rounded-xl border border-border bg-card p-6">
              <div className="mb-6 flex items-center gap-2">
                <Clock className="h-5 w-5 text-muted-foreground" />
                <h3 className="text-lg font-semibold text-foreground">Time & Data Configuration</h3>
              </div>
              <div className="space-y-6">
                <SliderParameter
                  label="Time Horizon"
                  value={parameters.timeHorizon}
                  onChange={(value) => updateParameter("timeHorizon", value)}
                  min={21}
                  max={1260}
                  step={21}
                  formatValue={(v) => `${v} days (~${Math.round(v / 252)} yr)`}
                />
                <SliderParameter
                  label="Data Points"
                  value={parameters.dataPoints}
                  onChange={(value) => updateParameter("dataPoints", value)}
                  min={100}
                  max={10000}
                  step={100}
                  formatValue={(v) => v.toLocaleString()}
                />
              </div>
            </section>
          </div>
        )

      case 4:
        return (
          <div className="space-y-8">
            <section className="rounded-xl border border-border bg-card p-6">
              <div className="mb-6 flex items-center gap-2">
                <Sliders className="h-5 w-5 text-muted-foreground" />
                <h3 className="text-lg font-semibold text-foreground">Statistical Properties</h3>
                <InfoTooltip content="Fine-tune the statistical characteristics of your generated data." />
              </div>
              <div className="space-y-3">
                <ToggleParameter
                  label="Mean Reversion"
                  description="Prices tend to return to a long-term average"
                  checked={parameters.meanReversion}
                  onChange={(checked) => updateParameter("meanReversion", checked)}
                />
                <ToggleParameter
                  label="Fat Tails"
                  description="Higher probability of extreme price movements"
                  checked={parameters.fatTails}
                  onChange={(checked) => updateParameter("fatTails", checked)}
                />
                <ToggleParameter
                  label="Jump Diffusion"
                  description="Include sudden large price jumps"
                  checked={parameters.jumpDiffusion}
                  onChange={(checked) => updateParameter("jumpDiffusion", checked)}
                />
              </div>
            </section>

            <section className="rounded-xl border border-border bg-card p-6">
              <div className="mb-6 flex items-center gap-2">
                <Layers className="h-5 w-5 text-muted-foreground" />
                <h3 className="text-lg font-semibold text-foreground">Correlation Settings</h3>
              </div>
              <SliderParameter
                label="Correlation Strength"
                value={parameters.correlationStrength}
                onChange={(value) => updateParameter("correlationStrength", value)}
                min={-1}
                max={1}
                step={0.1}
                formatValue={(v) => {
                  if (v > 0.5) return `${v.toFixed(1)} (Strong +)`
                  if (v > 0) return `${v.toFixed(1)} (Weak +)`
                  if (v === 0) return `${v.toFixed(1)} (None)`
                  if (v > -0.5) return `${v.toFixed(1)} (Weak -)`
                  return `${v.toFixed(1)} (Strong -)`
                }}
              />
            </section>

            <div className="rounded-xl border border-primary/30 bg-primary/5 p-4">
              <p className="text-sm text-primary">
                Ready to generate! Review your configuration in the sidebar and click
                &quot;Generate Data&quot; when ready.
              </p>
            </div>
          </div>
        )

      default:
        return null
    }
  }

  return (
    <div className="space-y-6">
      <div className="rounded-xl border border-border bg-card p-4">
        <StepIndicator
          steps={steps}
          currentStep={currentStep}
          onStepClick={setCurrentStep}
        />
      </div>

      <div className="grid gap-6 lg:grid-cols-[1fr_320px]">
        <div className="min-h-[500px]">{renderStepContent()}</div>

        <div className="lg:block">
          <ConfigSummary
            parameters={parameters}
            onGenerate={handleGenerate}
            onReset={handleReset}
            isGenerating={isGenerating}
            currentStep={currentStep}
            totalSteps={steps.length}
            onNext={handleNext}
            onBack={handleBack}
            canProceed={true}
          />
        </div>
      </div>
    </div>
  )
}
