"use client"

import { useState } from "react"
import { Header } from "@/components/header"
import { ParameterizationForm } from "@/components/parameterization-form"
import { VisualizationResults } from "@/components/graphs-results"
import { type GeneratedData } from "@/lib/types"

export default function Home() {
  const [generatedData, setGeneratedData] = useState<GeneratedData | null>(null)

  return (
    <div className="min-h-screen bg-background">
      <Header />
      <main className="container mx-auto max-w-7xl px-4 py-6 sm:py-8">
        <div className="mb-6">
          <h2 className="text-xl font-bold tracking-tight text-foreground sm:text-2xl">
            Configure Your Parameters
          </h2>
          <p className="mt-1 text-sm text-muted-foreground sm:text-base">
            Set your model parameters and upload a time series CSV to generate synthetic data.
          </p>
        </div>
        <ParameterizationForm onDataReady={setGeneratedData} />
        {generatedData && (
          <div className="mt-10">
            <h2 className="mb-6 text-xl font-bold tracking-tight text-foreground sm:text-2xl">
              Results
            </h2>
            <VisualizationResults data={generatedData} />
          </div>
        )}
      </main>
    </div>
  )
}
