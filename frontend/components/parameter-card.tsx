"use client"

import { cn } from "@/lib/utils"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import type { ReactNode } from "react"

interface ParameterCardProps {
  title: string
  description: string
  icon: ReactNode
  children: ReactNode
  className?: string
}

export function ParameterCard({ title, description, icon, children, className }: ParameterCardProps) {
  return (
    <Card className={cn("bg-card border-border", className)}>
      <CardHeader className="pb-4">
        <div className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-secondary text-muted-foreground">
            {icon}
          </div>
          <div>
            <CardTitle className="text-base font-medium text-foreground">{title}</CardTitle>
            <CardDescription className="text-sm text-muted-foreground">{description}</CardDescription>
          </div>
        </div>
      </CardHeader>
      <CardContent>{children}</CardContent>
    </Card>
  )
}
