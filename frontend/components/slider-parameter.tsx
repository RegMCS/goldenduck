"use client"

import { Slider } from "@/components/ui/slider"
import { cn } from "@/lib/utils"

interface SliderParameterProps {
  label: string
  value: number
  onChange: (value: number) => void
  min: number
  max: number
  step?: number
  unit?: string
  formatValue?: (value: number) => string
  className?: string
}

export function SliderParameter({
  label,
  value,
  onChange,
  min,
  max,
  step = 1,
  unit = "",
  formatValue,
  className,
}: SliderParameterProps) {
  const displayValue = formatValue ? formatValue(value) : `${value}${unit}`

  return (
    <div className={cn("space-y-3", className)}>
      <div className="flex items-center justify-between">
        <span className="text-sm font-medium text-foreground">{label}</span>
        <span className="rounded-md bg-secondary px-2 py-1 font-mono text-sm text-primary">
          {displayValue}
        </span>
      </div>
      <Slider
        value={[value]}
        onValueChange={([v]) => onChange(v)}
        min={min}
        max={max}
        step={step}
        className="w-full"
      />
      <div className="flex justify-between text-xs text-muted-foreground">
        <span>{formatValue ? formatValue(min) : `${min}${unit}`}</span>
        <span>{formatValue ? formatValue(max) : `${max}${unit}`}</span>
      </div>
    </div>
  )
}
