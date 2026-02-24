export interface MarketParameters {
  volatility: number // 0.0 to 1.0
  trend: number // -1 to +1
  fatTails: number // 0.0 to 1.0
  momentum: number // 0 to 1
  timeHorizon: number // 60 to 500 days
  inputFile: File | null // CSV with OHLCV headers
}

export const defaultParameters: MarketParameters = {
  volatility: 1.0,
  trend: 0,
  fatTails: 1.0,
  momentum: 0.5,
  timeHorizon: 252,
  inputFile: null,
}

export const REQUIRED_CSV_HEADERS = ["open", "high", "low", "close", "volume"]
