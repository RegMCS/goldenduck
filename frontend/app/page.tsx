import { Header } from "@/components/header"
import { ParameterizationForm } from "@/components/parameterization-form"

export default function Home() {
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
        <ParameterizationForm />
      </main>
    </div>
  )
}
