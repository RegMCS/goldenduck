import argparse
import os
from worker.DDPM.services.inference      import InferenceService
from worker.DDPM.services.reconstruction import ReconstructionService
from worker.DDPM.services.evaluate       import EvaluationService


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint",   default="worker/DDPM/checkpoints/stage2_finetuned.pt")
    ap.add_argument("--artefact_dir", default="worker/DDPM/artefacts")
    ap.add_argument("--ohlcv_path",   required=True)
    ap.add_argument("--regime",       default="crisis",
                    choices=["calm", "highvol", "crisis", "custom"])
    ap.add_argument("--n_paths",      type=int,   default=100)
    ap.add_argument("--guidance",     type=float, default=3.0)
    ap.add_argument("--asset_idx",    type=int,   default=0)
    ap.add_argument("--out_file",     default=None,
                    help="Override output CSV path (optional, auto-named if not set)")
    ap.add_argument("--eval_dir",     default=None,
                    help="Override eval output dir (optional, auto-named if not set)")

    # Custom conditioning knobs
    ap.add_argument("--realised_vol", type=float, default=None)
    ap.add_argument("--drift",        type=float, default=None)
    ap.add_argument("--tail_index",   type=float, default=None)
    ap.add_argument("--momentum",     type=float, default=None)

    args = ap.parse_args()

    # ── Validate custom regime ────────────────────────────────────────────
    if args.regime == "custom":
        missing = [p for p in ["realised_vol", "drift", "tail_index", "momentum"]
                   if getattr(args, p) is None]
        if missing:
            ap.error(f"--regime=custom requires: {', '.join('--' + m for m in missing)}")
        custom_cond = {
            "realised_vol": args.realised_vol,
            "drift":        args.drift,
            "tail_index":   args.tail_index,
            "momentum":     args.momentum,
        }
    else:
        custom_cond = None

    # ── Auto-name output paths ────────────────────────────────────────────
    os.makedirs("worker/DDPM/output", exist_ok=True)

    if args.out_file:
        out_file = args.out_file
    elif args.regime == "custom":
        tag = (f"custom"
               f"_vol-{args.realised_vol}"
               f"_drift-{args.drift}"
               f"_tail-{args.tail_index}"
               f"_mom-{args.momentum}")
        out_file = f"worker/DDPM/output/synthetic_{tag}.csv"
    else:
        tag = (f"regime-{args.regime}"
               f"_guidance-{args.guidance}"
               f"_npaths-{args.n_paths}")
        out_file = f"worker/DDPM/output/synthetic_{tag}.csv"

    if args.eval_dir:
        eval_dir = args.eval_dir
    elif args.regime == "custom":
        eval_dir = (f"worker/DDPM/output/eval"
                    f"/custom_vol-{args.realised_vol}"
                    f"_drift-{args.drift}"
                    f"_tail-{args.tail_index}"
                    f"_mom-{args.momentum}")
    else:
        eval_dir = (f"worker/DDPM/output/eval"
                    f"/regime-{args.regime}"
                    f"_guidance-{args.guidance}")

    # ── Run inference ─────────────────────────────────────────────────────
    svc   = InferenceService(args.checkpoint, args.artefact_dir)
    paths = svc.generate(
        regime         = args.regime,
        n_paths        = args.n_paths,
        guidance_scale = args.guidance,
        asset_idx      = args.asset_idx,
        custom_cond    = custom_cond,
    )
    print(f"Generated paths: {paths.shape}")

    # ── Reconstruct + export CSV ──────────────────────────────────────────
    rec = ReconstructionService(args.ohlcv_path)
    dfs = rec.reconstruct(paths, regime=args.regime)
    rec.export_csv(dfs, out_file)
    print(f"Saved synthetic CSV → {out_file}")

    # ── Evaluate ──────────────────────────────────────────────────────────
    eval_svc = EvaluationService(args.ohlcv_path)
    eval_svc.evaluate(out_file, out_dir=eval_dir)
    print(f"Evaluation saved  → {eval_dir}")


if __name__ == "__main__":
    main()