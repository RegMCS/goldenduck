import { render, screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { describe, expect, it, vi } from "vitest"
import { VisualizationResults } from "@/components/graphs-results"
import type { GeneratedData } from "@/lib/types"

// Mock the chart components since they're complex and might have their own tests
vi.mock("@/components/charts/candlestick-chart", () => ({
  CandlestickChart: ({ title }: { title: string }) => <div data-testid="candlestick-chart">{title}</div>,
}))

vi.mock("@/components/charts/price-overlay-chart", () => ({
  PriceOverlayChart: () => <div data-testid="price-overlay-chart">Price Overlay Chart</div>,
}))

vi.mock("@/components/charts/cumulative-return-chart", () => ({
  CumulativeReturnChart: () => <div data-testid="cumulative-return-chart">Cumulative Return Chart</div>,
}))

vi.mock("@/components/charts/drawdown-chart", () => ({
  DrawdownChart: () => <div data-testid="drawdown-chart">Drawdown Chart</div>,
}))

vi.mock("@/components/charts/stats-panel", () => ({
  StatsPanel: () => <div data-testid="stats-panel">Stats Panel</div>,
}))

// Mock Tabs component
vi.mock("@/components/ui/tabs", () => ({
  Tabs: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  TabsContent: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  TabsList: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  TabsTrigger: ({ children }: { children: React.ReactNode }) => <button>{children}</button>,
}))

const mockData: GeneratedData = {
  historical: [
    { date: "2024-01-01", open: 100, high: 110, low: 90, close: 105, volume: 1000 },
    { date: "2024-01-02", open: 105, high: 115, low: 100, close: 110, volume: 1200 },
  ],
  synthetic: [
    { date: "2024-01-01", open: 100, high: 110, low: 90, close: 105, volume: 1000 },
    { date: "2024-01-02", open: 105, high: 115, low: 100, close: 110, volume: 1200 },
  ],
  timeSeries: [
    { date: "2024-01-01", timestamp: 1704067200, historical: 100, synthetic: 105 },
    { date: "2024-01-02", timestamp: 1704153600, historical: 105, synthetic: 110 },
  ],
  returns: [
    { date: "2024-01-01", timestamp: 1704067200, historicalReturn: 0.05, syntheticReturn: 0.06, historicalCumReturn: 0.05, syntheticCumReturn: 0.06 },
    { date: "2024-01-02", timestamp: 1704153600, historicalReturn: 0.03, syntheticReturn: 0.04, historicalCumReturn: 0.08, syntheticCumReturn: 0.10 },
  ],
  drawdowns: [
    { date: "2024-01-01", timestamp: 1704067200, historicalDrawdown: -0.02, syntheticDrawdown: -0.01 },
    { date: "2024-01-02", timestamp: 1704153600, historicalDrawdown: -0.01, syntheticDrawdown: -0.005 },
  ],
  stats: {
    historical: {
      mean: 107.5,
      std: 3.54,
      skewness: 0.1,
      kurtosis: 3.2,
      maxDrawdown: -0.15,
      sharpe: 0.67,
      annualizedReturn: 0.12,
      annualizedVol: 0.18,
      totalReturn: 0.1,
      numDataPoints: 2,
    },
    synthetic: {
      mean: 107.5,
      std: 3.54,
      skewness: 0.2,
      kurtosis: 2.8,
      maxDrawdown: -0.12,
      sharpe: 0.88,
      annualizedReturn: 0.14,
      annualizedVol: 0.16,
      totalReturn: 0.1,
      numDataPoints: 2,
    },
  },
}

describe("VisualizationResults", () => {
  it("renders all main sections with data", () => {
    render(<VisualizationResults data={mockData} />)

    // Check main sections are rendered
    expect(screen.getByText("OHLCV Candlestick Comparison")).toBeInTheDocument()
    expect(screen.getByText("Analytics")).toBeInTheDocument()
    expect(screen.getByText("Statistical Comparison")).toBeInTheDocument()
    
    // Check that chart components are rendered
    expect(screen.getAllByTestId("candlestick-chart").length).toBeGreaterThan(0)
    expect(screen.getByTestId("stats-panel")).toBeInTheDocument()
  })

  it("displays fidelity gauge with score", () => {
    render(<VisualizationResults data={mockData} />)
    
    expect(screen.getByText("Fidelity")).toBeInTheDocument()
    expect(screen.getByText("score")).toBeInTheDocument()
    expect(screen.getByText("vs. historical")).toBeInTheDocument()
  })

  it("shows dual metric cards with historical and synthetic values", () => {
    render(<VisualizationResults data={mockData} />)
    
    // Check that metric cards are displayed
    expect(screen.getByText("Ann. Return")).toBeInTheDocument()
    expect(screen.getByText("Ann. Volatility")).toBeInTheDocument()
    expect(screen.getByText("Sharpe Ratio")).toBeInTheDocument()
    expect(screen.getByText("Max Drawdown")).toBeInTheDocument()
    
    // Check that values are displayed
    expect(screen.getByText("12.00%")).toBeInTheDocument() // historical annual return
    expect(screen.getByText("14.00%")).toBeInTheDocument() // synthetic annual return
  })

  it("renders analytics tabs with correct options", () => {
    render(<VisualizationResults data={mockData} />)
    
    expect(screen.getByText("Price Overlay")).toBeInTheDocument()
    expect(screen.getByText("Cum. Returns")).toBeInTheDocument()
    expect(screen.getByText("Drawdowns")).toBeInTheDocument()
  })

  it("handles empty data gracefully", () => {
    const emptyData: GeneratedData = {
      historical: [],
      synthetic: [],
      timeSeries: [],
      returns: [],
      drawdowns: [],
      stats: {
        historical: {
          mean: 0,
          std: 0,
          skewness: 0,
          kurtosis: 0,
          maxDrawdown: 0,
          sharpe: 0,
          annualizedReturn: 0,
          annualizedVol: 0,
          totalReturn: 0,
          numDataPoints: 0,
        },
        synthetic: {
          mean: 0,
          std: 0,
          skewness: 0,
          kurtosis: 0,
          maxDrawdown: 0,
          sharpe: 0,
          annualizedReturn: 0,
          annualizedVol: 0,
          totalReturn: 0,
          numDataPoints: 0,
        },
      },
    }

    render(<VisualizationResults data={emptyData} />)
    
    // Should still render without crashing
    expect(screen.getByText("OHLCV Candlestick Comparison")).toBeInTheDocument()
    expect(screen.getByText("Analytics")).toBeInTheDocument()
  })
})