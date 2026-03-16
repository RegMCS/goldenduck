"use client"

import { Clock, AlertCircle } from "lucide-react"
import { type MarketParameters } from "@/lib/types"

interface TimingEstimateProps {
  parameters: MarketParameters
  className?: string
}

// Timing estimation algorithm based on input parameters
// These are empirical estimates based on typical GARCH model performance
function estimateJobDuration(parameters: MarketParameters): {
  baseSeconds: number
  totalSeconds: number
  breakdown: {
    label: string
    seconds: number
  }[]
} {
  // Base time for data loading and preprocessing
  const baseProcessingTime = 5
  
  // Time factors based on parameters
  const volatilityFactor = Math.max(0.5, parameters.volatility) // Higher volatility = more complex modeling
  const fatTailsFactor = Math.max(0.8, parameters.fatTails) // Fat tails require more iterations
  const momentumFactor = Math.max(0.7, parameters.momentum) // Momentum increases model complexity
  const horizonFactor = parameters.timeHorizon / 252 // Scale with horizon (252 = typical trading year)
  
  // Model complexity estimation
  const modelComplexity = volatilityFactor * fatTailsFactor * momentumFactor
  
  // Time breakdown estimation
  const breakdown = [
    { label: "Data loading", seconds: 2 },
    { label: "Model fitting", seconds: baseProcessingTime * modelComplexity },
    { label: "Scenario generation", seconds: 3 * horizonFactor },
    { label: "Validation", seconds: 2 },
  ]
  
  const totalSeconds = breakdown.reduce((sum, item) => sum + item.seconds, 0)
  
  return {
    baseSeconds: totalSeconds,
    totalSeconds: Math.max(5, Math.min(120, totalSeconds)), // Clamp between 5s and 2min
    breakdown
  }
}

function formatDuration(seconds: number): string {
  if (seconds < 60) {
    return `${Math.ceil(seconds)} seconds`
  } else {
    const minutes = Math.ceil(seconds / 60)
    return `${minutes} minute${minutes > 1 ? 's' : ''}`
  }
}

export function TimingEstimate({ parameters, className = "" }: TimingEstimateProps) {
  const { inputFile, ...timingParams } = parameters
  const hasFile = !!inputFile
  
  const duration = estimateJobDuration(parameters)
  
  if (!hasFile) {
    return (
      <div className={`rounded-lg border border-amber-500/40 bg-amber-50/60 dark:bg-amber-950/20 px-4 py-3 ${className}`}>
        <div className="flex items-center gap-2">
          <AlertCircle className="h-4 w-4 text-amber-600 dark:text-amber-400" />
          <span className="text-sm font-medium text-amber-700 dark:text-amber-400">
            Upload CSV to see timing estimate
          </span>
        </div>
      </div>
    )
  }
  
  return (
    <div className={`rounded-lg border border-border bg-muted/50 px-4 py-3 ${className}`}>
      <div className="flex items-center gap-2 mb-2">
        <Clock className="h-4 w-4 text-blue-600 dark:text-blue-400" />
        <span className="text-sm font-medium text-foreground">
          Estimated Processing Time
        </span>
      </div>
      
      <div className="space-y-1.5">
        <div className="flex justify-between items-center">
          <span className="text-sm text-muted-foreground">Total</span>
          <span className="text-sm font-semibold text-foreground">
            {formatDuration(duration.totalSeconds)}
          </span>
        </div>
        
        <div className="text-xs text-muted-foreground space-y-0.5">
          {duration.breakdown.map((item, index) => (
            <div key={index} className="flex justify-between">
              <span>{item.label}</span>
              <span>{Math.ceil(item.seconds)}s</span>
            </div>
          ))}
        </div>
        
        <div className="pt-1 text-xs text-muted-foreground">
          <span className="inline-block h-1 bg-blue-200 dark:bg-blue-800 rounded-full w-full">
            <span 
              className="block h-1 bg-blue-600 dark:bg-blue-400 rounded-full transition-all duration-300"
              style={{ width: `${Math.min(100, (duration.totalSeconds / 120) * 100)}%` }}
            />
          </span>
        </div>
        
        <div className="text-xs text-muted-foreground italic">
          Time varies based on parameter complexity
        </div>
      </div>
    </div>
  )
}