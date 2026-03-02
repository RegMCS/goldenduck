"use client"

import { Zap } from "lucide-react"
import { Badge } from "@/components/ui/badge"
import { presets, type Preset } from "@/lib/presets"
import type { PresetMarketParameters } from "@/lib/types"

interface PresetSelectorProps {
  onSelect: (preset: Preset) => void
  currentParams: PresetMarketParameters
}

export function PresetSelector({ onSelect, currentParams }: PresetSelectorProps) {
  const isPresetActive = (preset: Preset) => {
    return Object.entries(preset.parameters).every(
      ([key, value]) => currentParams[key as keyof PresetMarketParameters] === value
    )
  }

  return (
    <div className="rounded-xl border border-border bg-card p-4">
      <div className="mb-3 flex items-center gap-2">
        <Zap className="h-4 w-4 text-warning" />
        <span className="text-sm font-medium text-foreground">Quick Presets</span>
      </div>
      <div className="flex flex-wrap gap-2">
        {presets.map((preset) => {
          const isActive = isPresetActive(preset)
          return (
            <button
              key={preset.id}
              type="button"
              onClick={() => onSelect(preset)}
              className="group"
            >
              <Badge
                variant={isActive ? "default" : "outline"}
                className={
                  isActive
                    ? "bg-primary text-primary-foreground"
                    : "border-border bg-secondary text-muted-foreground hover:bg-muted hover:text-foreground"
                }
              >
                {preset.name}
              </Badge>
            </button>
          )
        })}
      </div>
      <p className="mt-2 text-xs text-muted-foreground">
        Click a preset to quickly configure common market scenarios
      </p>
    </div>
  )
}
