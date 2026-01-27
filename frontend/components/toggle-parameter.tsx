"use client"

import { Switch } from "@/components/ui/switch"
import { cn } from "@/lib/utils"

interface ToggleParameterProps {
  label: string
  description: string
  checked: boolean
  onChange: (checked: boolean) => void
  className?: string
}

export function ToggleParameter({
  label,
  description,
  checked,
  onChange,
  className,
}: ToggleParameterProps) {
  return (
    <div
      className={cn(
        "flex items-center justify-between rounded-lg border border-border bg-secondary/50 p-4",
        className
      )}
    >
      <div className="space-y-0.5">
        <label className="text-sm font-medium text-foreground">{label}</label>
        <p className="text-xs text-muted-foreground">{description}</p>
      </div>
      <Switch checked={checked} onCheckedChange={onChange} />
    </div>
  )
}
