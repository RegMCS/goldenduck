
import argparse
from worker.DDPM.services.inference      import InferenceService
from worker.DDPM.services.reconstruction import ReconstructionService


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint",   default="checkpoints/stage2_finetuned.pt")
    ap.add_argument("--artefact_dir", default="artefacts")
    ap.add_argument("--ohlcv_path",   required=True)
    ap.add_argument("--regime",       default="crisis",
                    choices=["calm", "highvol", "crisis"])
    ap.add_argument("--n_paths",      type=int,   default=100)
    ap.add_argument("--guidance",     type=float, default=3.0)
    ap.add_argument("--asset_idx",    type=int,   default=0)
    ap.add_argument("--out_file",     default="output/synthetic_ohlcv.csv")
    args = ap.parse_args()

    svc   = InferenceService(args.checkpoint, args.artefact_dir)
    paths = svc.generate(
        regime         = args.regime,
        n_paths        = args.n_paths,
        guidance_scale = args.guidance,
        asset_idx      = args.asset_idx,
    )
    print(f"Generated paths: {paths.shape}")

    rec = ReconstructionService(args.ohlcv_path)
    dfs = rec.reconstruct(paths, regime=args.regime)
    rec.export_csv(dfs, args.out_file)


if __name__ == "__main__":
    main()