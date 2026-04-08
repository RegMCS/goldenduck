"use client"

import { ActiveModelTable } from "@/components/training/active-model-table"

interface Props {
  tableRefreshTrigger: number
}

/** Unified list of upload + trained artifacts; activate any row. */
export function ModelActivationDashboard({ tableRefreshTrigger }: Props) {
  return <ActiveModelTable refreshTrigger={tableRefreshTrigger} />
}
