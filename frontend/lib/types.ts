export interface MarketParameters {
  volatility: number // 0.5 to 2.0 (multiplier)
  trend: number // -1.0 to +1.0 (directional)
  fatTails: number // 0.5 to 2.0 (multiplier)
  momentum: number // 0.0 to 1.0 (probability-like)
  timeHorizon: number // 500 to 2600 days
  inputFile: File | null // CSV with OHLCV headers
}

export const defaultParameters: MarketParameters = {
  volatility: 1.0,
  trend: 0,
  fatTails: 1.0,
  momentum: 0.5,
  timeHorizon: 500,
  inputFile: null,
}

export interface PresetMarketParameters {
  assetClass: "equities" | "fx" | "crypto"
  trendType: "bearish" | "bullish" | "sideways" | "volatile"
  marketRegime: "crisis" | "normal" | "low-volatility" | "high-volatility"
  volatilityLevel: "low" | "medium" | "extreme"
  generationModel: string
  timeHorizon: number
  meanReversion: boolean
  fatTails: boolean
  jumpDiffusion: boolean
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
  overallMatch?: number
}

export const REQUIRED_CSV_HEADERS = ["open", "high", "low", "close", "volume"]

export type JobStatus = "queued" | "running" | "completed" | "failed"
export type JobType = "garch" | "gan" | "ddpm"

export interface JobHistoryItem {
  id: string
  status: JobStatus
  job_type: JobType | null
  requested_at: string
  completed_at: string | null
  s3_url: string | null
}

export interface JobHistoryResponse {
  jobs: JobHistoryItem[]
  total: number
  total_completed: number
  total_running: number
  total_failed: number
  total_queued: number
}