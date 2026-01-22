from datetime import datetime
import os


def generate_markdown_report(all_results: dict, output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as f:

        f.write("# GARCH Evaluation Report\n\n")
        f.write(f"Generated on: {datetime.now()}\n\n")

        for ticker, dist_results in all_results.items():
            f.write(f"## {ticker}\n\n")

            for dist, result in dist_results.items():
                if "error" in result:
                    f.write(f"### {dist}\n")
                    f.write(f"- Error: {result['error']}\n\n")
                    continue

                params = result["parameters"]
                metrics = result["metrics"]

                f.write(f"### Distribution: {dist}\n\n")
                f.write("**Model Parameters**\n")
                f.write(f"- Converged: {params['converged']}\n")
                f.write(f"- AIC: {params['aic']:.4f}\n")
                f.write(f"- BIC: {params['bic']:.4f}\n")
                f.write(f"- Alpha: {params['alpha']:.4f}\n")
                f.write(f"- Beta: {params['beta']:.4f}\n")
                f.write(f"- Omega: {params['omega']:.4f}\n\n")

                f.write("**Validation Metrics**\n")
                f.write(f"- KS statistic: {metrics['ks_statistic']:.4f}\n")
                f.write(f"- KS p-value: {metrics['ks_pvalue']:.4f}\n")
                f.write(
                    f"- Kurtosis (historical): {metrics['kurtosis_historical']:.4f}\n"
                )
                f.write(
                    f"- Kurtosis (synthetic): {metrics['kurtosis_synthetic']:.4f}\n"
                )
                f.write(
                    f"- Skewness (historical): {metrics['skewness_historical']:.4f}\n"
                )
                f.write(
                    f"- Skewness (synthetic): {metrics['skewness_synthetic']:.4f}\n"
                )
                f.write(
                    f"- ACF lag-1 (historical): {metrics['acf_lag1_historical']:.4f}\n"
                )
                f.write(
                    f"- ACF lag-1 (synthetic): {metrics['acf_lag1_synthetic']:.4f}\n\n"
                )

            f.write("---\n\n")
