"use client"

import { Info } from "lucide-react"
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip"

interface InfoTooltipProps {
  content: string | string[]
}

export function InfoTooltip({ content }: InfoTooltipProps) {
  return (
    <TooltipProvider delayDuration={200}>
      <Tooltip>
        <TooltipTrigger asChild>
          <span
            role="button"
            tabIndex={0}
            aria-label="More info"
            className="inline-flex items-center justify-center rounded-full text-muted-foreground transition-colors hover:text-foreground focus:outline-none focus-visible:ring-2 focus-visible:ring-ring"
          >
            <Info className="h-4 w-4" />
            <span className="sr-only">More info</span>
          </span>
        </TooltipTrigger>

        <TooltipContent
          side="top"
          className="max-w-[280px] bg-popover text-popover-foreground"
        >
          {Array.isArray(content) ? (
            <div className="space-y-1.5">
              <p className="text-sm font-medium">{content[0]}</p>
              <ul className="space-y-1 text-sm list-disc pl-3">
                {content.slice(1).filter((item) => !item.startsWith("Note:")).map((item) => (
                  <li key={item}>{item}</li>
                ))}
              </ul>
              {content.slice(1).filter((item) => item.startsWith("Note:")).map((item) => (
                <p key={item} className="text-xs text-muted-foreground italic">{item}</p>
              ))}
            </div>
          ) : (
            <p className="text-sm">{content}</p>
          )}
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  )
}
