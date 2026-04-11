"""
reconstruction.py
-----------------
Converts generated log-return paths to OHLCV DataFrames and exports to CSV.
"""
import os
import numpy as np
import pandas as pd
from worker.DDPM.utils import load_ohlcv, returns_to_ohlcv


class ReconstructionService:
    def __init__(self, ohlcv_path: str):
        self.df_input, self.first_close = load_ohlcv(ohlcv_path)

    def reconstruct(self, log_returns: np.ndarray,
                    regime: str = "custom") -> list:
        """
        log_returns : (n_paths, seq_len)
        Returns list of n_paths OHLCV DataFrames.
        """
        dfs = []
        for i, path in enumerate(log_returns):
            df = returns_to_ohlcv(path, self.first_close, self.df_input)
            df["path_id"] = i
            df["regime"]  = regime
            dfs.append(df)
        return dfs

    def export_csv(self, dfs: list,
                   out_path: str = "synthetic_ohlcv.csv") -> pd.DataFrame:
        df_all = pd.concat(dfs)
        df_all.index.name = "Date"
        out_dir = os.path.dirname(out_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        df_all.to_csv(out_path)
        print(f"Exported {len(dfs)} paths to {out_path}  shape={df_all.shape}")
        return df_all

    def export_excel(self, dfs: list,
                     out_path: str = "synthetic_ohlcv.xlsx"):
        """One sheet per path (max 20 paths)."""
        subset = dfs[:20]
        with pd.ExcelWriter(out_path) as writer:
            for i, df in enumerate(subset):
                df.drop(columns=["path_id", "regime"], errors="ignore") \
                  .to_excel(writer, sheet_name=f"path_{i + 1}")
        print(f"Exported Excel to {out_path}")