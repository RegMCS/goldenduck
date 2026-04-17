import os
import subprocess

BASELINE = {
    "realised_vol": 0.10,
    "drift":        0.05,
    "tail_index":   0.008,
    "momentum":    -0.05,
}

SCENARIOS = [
    ("baseline",           {**BASELINE}),
    ("realised_vol_low",   {**BASELINE, "realised_vol": 0.10}),
    ("realised_vol_mid",   {**BASELINE, "realised_vol": 0.30}),
    ("realised_vol_high",  {**BASELINE, "realised_vol": 0.55}),
    ("drift_low",          {**BASELINE, "drift":  0.05}),
    ("drift_mid",          {**BASELINE, "drift": -0.05}),
    ("drift_high",         {**BASELINE, "drift": -0.25}),
    ("tail_index_low",     {**BASELINE, "tail_index": 0.008}),
    ("tail_index_mid",     {**BASELINE, "tail_index": 0.025}),
    ("tail_index_high",    {**BASELINE, "tail_index": 0.055}),
    ("momentum_low",       {**BASELINE, "momentum": -0.05}),
    ("momentum_mid",       {**BASELINE, "momentum": -0.10}),
    ("momentum_high",      {**BASELINE, "momentum": -0.15}),
]

os.makedirs("worker/DDPM/output", exist_ok=True)

total = len(SCENARIOS)
print(f"Running {total} scenarios...\n")

for i, (label, cond) in enumerate(SCENARIOS, 1):
    print(f"[{i}/{total}] {label}")
    cmd = [
        "python", "-m", "worker.DDPM.scripts.run_inference",
        "--ohlcv_path",   "worker/DDPM/data/AAPL.csv",
        "--regime",       "custom",
        "--out_file",     f"worker/DDPM/output/synthetic_{label}.csv",
        "--eval_dir",     f"worker/DDPM/output/eval/{label}",
        "--realised_vol", str(cond["realised_vol"]),
        "--drift",        str(cond["drift"]),
        "--tail_index",   str(cond["tail_index"]),
        "--momentum",     str(cond["momentum"]),
    ]
    result = subprocess.run(cmd)
    print(f"  {'✅ Done' if result.returncode == 0 else '❌ FAILED'}\n")

print("Batch complete.")
print("CSVs saved to worker/DDPM/output/")