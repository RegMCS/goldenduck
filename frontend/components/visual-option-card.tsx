"use client"

import type { ReactNode } from "react"
import { cn } from "@/lib/utils"
import { Check } from "lucide-react"

interface VisualOptionCardProps {
  icon: ReactNode
  label: string
  description: string
  selected: boolean
  onClick: () => void
  accentColor?: string
}

export function VisualOptionCard({
  icon,
  label,
  description,
  selected,
  onClick,
  accentColor,
}: VisualOptionCardProps) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={cn(
        "group relative flex flex-col items-center gap-2 rounded-xl border-2 p-4 text-center transition-all duration-200",
        selected
          ? "border-primary bg-primary/10 shadow-[0_0_20px_-5px_var(--primary)]"
          : "border-transparent bg-secondary/50 hover:bg-secondary hover:border-border"
      )}
      style={selected && accentColor ? { borderColor: accentColor, boxShadow: `0 0 20px -5px ${accentColor}` } : undefined}
    >
      {selected && (
        <div className="absolute top-2 right-2 flex h-5 w-5 items-center justify-center rounded-full bg-primary">
          <Check className="h-3 w-3 text-primary-foreground" />
        </div>
      )}
      <div
        className={cn(
          "flex h-12 w-12 items-center justify-center rounded-xl transition-all",
          selected ? "bg-primary/20 text-primary" : "bg-muted text-muted-foreground group-hover:text-foreground"
        )}
      >
        {icon}
      </div>
      <div className="space-y-0.5">
        <p className={cn("text-sm font-semibold", selected ? "text-primary" : "text-foreground")}>
          {label}
        </p>
        <p className="text-xs text-muted-foreground line-clamp-2">{description}</p>
      </div>
    </button>
  )
}
