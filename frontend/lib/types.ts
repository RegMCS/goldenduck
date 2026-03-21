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

export interface JobParameters {
  volatility: number
  trend: number
  fatTails: number
  momentum: number
  timeHorizon: number
  fileName: string | null
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

export type TrainingStatus = "queued" | "running" | "completed" | "failed"

export interface TrainingRunConfig {
  testing_mode: boolean
  n_assets: number
  n_scenarios: number
  run_evaluation: boolean
}

export interface DirectMetrics {
  rmse: number
  mae: number
  r2: number
  mape?: number
}

export interface BaselineComparison {
  ai_score: number
  baseline_fixed: number
  baseline_heuristic: number
  p_value_vs_fixed: number
  p_value_vs_heuristic: number
  significant_vs_fixed: boolean
  significant_vs_heuristic: boolean
}

export interface EvaluationReport {
  timestamp: string
  dataset_size: { train: number; val: number; test: number }
  direct_metrics: { delta: DirectMetrics }
  end_to_end_metrics: { mean_score: number }
  baseline_comparison: BaselineComparison
  quality_checks: Record<string, boolean>
  overall_pass: boolean
}

export interface TrainingRun {
  id: string
  status: TrainingStatus
  config: TrainingRunConfig
  triggered_by: string
  started_at: string
  completed_at: string | null
  step?: string            // e.g. "Step 2 of 4: Generating samples"
  step_index?: number      // 1–4
  model_name?: string      // e.g. "rf_delta_20260402_143000.pkl"
  is_active?: boolean
  evaluation_report?: EvaluationReport | null
  error?: string | null
}

export interface TrainingRunsResponse {
  runs: TrainingRun[]
  total: number
}

export interface TrainingModel {
  name: string             // filename e.g. rf_delta_20260402_143000.pkl
  created_at: string
  size_bytes: number
  is_active: boolean
  run_id?: string
}

/** Row from GET /api/training/uploads (manual S3 uploads) */
export interface UploadedModelRow {
  id: string
  model_name: string
  s3_key: string
  uploaded_by: string
  created_at: string
  file_size_bytes: number | null
  is_active: boolean
}

export interface UploadedModelsListResponse {
  uploads: UploadedModelRow[]
  total: number
}