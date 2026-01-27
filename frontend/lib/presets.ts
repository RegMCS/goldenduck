import type { MarketParameters } from "./types"

export interface Preset {
  id: string
  name: string
  description: string
  parameters: Partial<MarketParameters>
}

export const presets: Preset[] = [
  {
    id: "stress-test-2008",
    name: "2008 Crisis Simulation",
    description: "Simulate conditions similar to the 2008 financial crisis",
    parameters: {
      assetClass: "equities",
      trendType: "bearish",
      marketRegime: "crisis",
      volatilityLevel: "extreme",
      fatTails: true,
      jumpDiffusion: true,
    },
  },
  {
    id: "bull-market",
    name: "Bull Market Rally",
    description: "Strong upward trending market with moderate volatility",
    parameters: {
      trendType: "bullish",
      marketRegime: "normal",
      volatilityLevel: "medium",
      meanReversion: false,
    },
  },
  {
    id: "fx-carry",
    name: "FX Carry Trade",
    description: "Stable FX environment for carry strategy testing",
    parameters: {
      assetClass: "fx",
      trendType: "sideways",
      marketRegime: "low-volatility",
      volatilityLevel: "low",
      meanReversion: true,
    },
  },
  {
    id: "crypto-volatile",
    name: "Crypto Volatility",
    description: "High volatility crypto market conditions",
    parameters: {
      assetClass: "crypto",
      trendType: "volatile",
      marketRegime: "high-volatility",
      volatilityLevel: "extreme",
      fatTails: true,
      jumpDiffusion: true,
    },
  },
]
