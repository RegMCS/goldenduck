export interface MarketParameters {
  volatility: number // 0.5 to 2.0 (multiplier)
  trend: number // -1.0 to +1.0 (directional)
  fatTails: number // 0.5 to 2.0 (multiplier)
  momentum: number // 0.0 to 1.0 (probability-like)
  timeHorizon: number // 60 to 1300 days
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

export interface OHLCVDataPoint {
  date: string
  open: number
  high: number
  low: number
  close: number
  volume: number
}

export interface ReturnPoint {
  date: string
  timestamp: number
  historicalReturn: number
  syntheticReturn: number
  historicalCumReturn: number
  syntheticCumReturn: number
}

export interface DrawdownPoint {
  date: string
  timestamp: number
  historicalDrawdown: number
  syntheticDrawdown: number
}

export interface TimeSeriesPoint {
  date: string
  timestamp: number
  historical: number
  synthetic: number
}

export interface DataStatistics {
  historical: SeriesStats
  synthetic: SeriesStats
}

export interface SeriesStats {
  mean: number
  std: number
  skewness: number
  kurtosis: number
  maxDrawdown: number
  sharpe: number
  annualizedReturn: number
  annualizedVol: number
  totalReturn: number
  numDataPoints: number
}

export interface GeneratedData {
  historical: OHLCVDataPoint[]
  synthetic: OHLCVDataPoint[]
  timeSeries: TimeSeriesPoint[]
  returns: ReturnPoint[]
  drawdowns: DrawdownPoint[]
  stats: DataStatistics
}

export const REQUIRED_CSV_HEADERS = ["open", "high", "low", "close", "volume"]