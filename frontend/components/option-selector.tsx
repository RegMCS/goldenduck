"use client"

import { cn } from "@/lib/utils"

interface Option<T extends string> {
  value: T
  label: string
  description: string
}

interface OptionSelectorProps<T extends string> {
  options: Option<T>[]
  value: T
  onChange: (value: T) => void
  columns?: 2 | 3 | 4 | 5
}

export function OptionSelector<T extends string>({
  options,
  value,
  onChange,
  columns = 2,
}: OptionSelectorProps<T>) {
  const gridCols = {
    2: "grid-cols-2",
    3: "grid-cols-3",
    4: "grid-cols-2 lg:grid-cols-4",
    5: "grid-cols-2 lg:grid-cols-5",
  }

  return (
    <div className={cn("grid gap-3", gridCols[columns])}>
      {options.map((option) => (
        <button
          key={option.value}
          type="button"
          onClick={() => onChange(option.value)}
          className={cn(
            "group relative flex flex-col items-start rounded-lg border p-4 text-left transition-all hover:border-primary/50",
            value === option.value
              ? "border-primary bg-primary/10"
              : "border-border bg-secondary/50 hover:bg-secondary"
          )}
        >
          <div className="flex w-full items-center justify-between">
            <span
              className={cn(
                "text-sm font-medium",
                value === option.value ? "text-primary" : "text-foreground"
              )}
            >
              {option.label}
            </span>
            <div
              className={cn(
                "h-4 w-4 rounded-full border-2 transition-all",
                value === option.value
                  ? "border-primary bg-primary"
                  : "border-muted-foreground"
              )}
            >
              {value === option.value && (
                <div className="flex h-full w-full items-center justify-center">
                  <div className="h-1.5 w-1.5 rounded-full bg-primary-foreground" />
                </div>
              )}
            </div>
          </div>
          <span className="mt-1 text-xs text-muted-foreground">{option.description}</span>
        </button>
      ))}
    </div>
  )
}
