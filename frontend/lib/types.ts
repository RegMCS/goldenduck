export type AssetClass = "equities" | "fx" | "derivatives" | "commodities" | "crypto"

export type TrendType = "bullish" | "bearish" | "sideways" | "volatile"

export type MarketRegime = "normal" | "crisis" | "high-volatility" | "low-volatility" | "recovery"

export type VolatilityLevel = "low" | "medium" | "high" | "extreme"

export type GenerationModel = "garch" | "gan" | "ddpm"

export interface MarketParameters {
  assetClass: AssetClass
  trendType: TrendType
  marketRegime: MarketRegime
  volatilityLevel: VolatilityLevel
  generationModel: GenerationModel
  timeHorizon: number // in days
  dataPoints: number
  correlationStrength: number // -1 to 1
  meanReversion: boolean
  fatTails: boolean
  jumpDiffusion: boolean
}

export const defaultParameters: MarketParameters = {
  assetClass: "equities",
  trendType: "sideways",
  marketRegime: "normal",
  volatilityLevel: "medium",
  generationModel: "garch",
  timeHorizon: 252,
  dataPoints: 1000,
  correlationStrength: 0,
  meanReversion: false,
  fatTails: true,
  jumpDiffusion: false,
}

export const assetClassOptions: { value: AssetClass; label: string; description: string }[] = [
  { value: "equities", label: "Equities", description: "Stock market indices and individual securities" },
  { value: "fx", label: "FX", description: "Foreign exchange currency pairs" },
  { value: "derivatives", label: "Derivatives", description: "Options, futures, and other derivatives" },
  { value: "commodities", label: "Commodities", description: "Precious metals, energy, agriculture" },
  { value: "crypto", label: "Crypto", description: "Cryptocurrency assets" },
]

export const trendTypeOptions: { value: TrendType; label: string; description: string }[] = [
  { value: "bullish", label: "Bullish", description: "Upward trending market" },
  { value: "bearish", label: "Bearish", description: "Downward trending market" },
  { value: "sideways", label: "Sideways", description: "Range-bound market" },
  { value: "volatile", label: "Volatile", description: "High volatility with no clear direction" },
]

export const marketRegimeOptions: { value: MarketRegime; label: string; description: string }[] = [
  { value: "normal", label: "Normal", description: "Typical market conditions" },
  { value: "crisis", label: "Crisis", description: "2008-level market stress" },
  { value: "high-volatility", label: "High Volatility", description: "VIX > 30 environment" },
  { value: "low-volatility", label: "Low Volatility", description: "Compressed volatility regime" },
  { value: "recovery", label: "Recovery", description: "Post-crisis recovery phase" },
]

export const volatilityLevelOptions: { value: VolatilityLevel; label: string; description: string }[] = [
  { value: "low", label: "Low", description: "Annualized vol < 10%" },
  { value: "medium", label: "Medium", description: "Annualized vol 10-20%" },
  { value: "high", label: "High", description: "Annualized vol 20-40%" },
  { value: "extreme", label: "Extreme", description: "Annualized vol > 40%" },
]

export const generationModelOptions: { value: GenerationModel; label: string; description: string }[] = [
  { value: "garch", label: "GARCH", description: "Traditional volatility clustering model" },
  { value: "gan", label: "Conditional GAN", description: "Deep generative adversarial network" },
  { value: "ddpm", label: "DDPM", description: "Denoising diffusion probabilistic model" },
]

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
