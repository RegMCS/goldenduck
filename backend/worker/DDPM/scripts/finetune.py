import os, copy, argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from worker.DDPM.scripts.model import ConditionalDenoiser
from worker.DDPM.scripts.prepare_data import slice_context_prediction
from worker.DDPM.utils import (get_noise_schedule, q_sample, tail_weighted_loss,
                                load_ohlcv, build_conditioning_matrix,
                                normalise_windows)

FT_EPOCHS   = 80
FT_PATIENCE = 12
FT_LR       = 5e-5
FT_BATCH    = 16
CONTEXT_LEN = 252
PRED_LEN    = 1260


def finetune(ohlcv_path: str,
             checkpoint: str,
             artefact_dir: str = "artefacts",
             out_dir: str      = "checkpoints",
             epochs: int       = FT_EPOCHS,
             device_str: str   = "auto"):

    os.makedirs(out_dir, exist_ok=True)
    device = torch.device(
        "cuda" if (device_str == "auto" and torch.cuda.is_available())
        else device_str if device_str != "auto" else "cpu"
    )
    print(f"Fine-tuning on {ohlcv_path}  |  Device: {device}")

    df, _ = load_ohlcv(ohlcv_path)
    returns_1d = df[["log_return"]]

    ctx_wins, pred_wins, _ = slice_context_prediction(
        returns_1d, CONTEXT_LEN, PRED_LEN, step=21)
    if len(pred_wins) == 0:
        raise RuntimeError("Not enough data for fine-tuning windows. "
                           "Provide at least 5 years of daily OHLCV data.")

    pred_wins      = pred_wins.squeeze(1)[:, np.newaxis, :]
    pred_wins_norm, _ = normalise_windows(pred_wins)

    C_raw, C_norm, _ = build_conditioning_matrix(pred_wins)

    X = torch.tensor(pred_wins_norm, dtype=torch.float32)
    C = torch.tensor(C_norm,         dtype=torch.float32)
    loader = DataLoader(TensorDataset(X, C),
                        batch_size=FT_BATCH, shuffle=True)

    ckpt     = torch.load(checkpoint, map_location=device)
    seq_len  = ckpt.get("seq_len",  PRED_LEN)
    n_assets = ckpt.get("n_assets", 1)

    model = ConditionalDenoiser(seq_len=seq_len, in_ch=n_assets, cond_dim=4).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded checkpoint: {checkpoint}")

    T = 200
    betas, alphas, alpha_bar = get_noise_schedule(T=T, device=str(device))

    opt   = torch.optim.AdamW(model.parameters(), lr=FT_LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best_loss    = float("inf")
    best_weights = None
    patience_ctr = 0

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for x0, c in loader:
            x0, c   = x0.to(device), c.to(device)
            noise   = torch.randn_like(x0)
            t       = torch.randint(0, T, (x0.shape[0],), device=device)
            x_noisy = q_sample(x0, t, noise, alpha_bar)
            pred    = model(x_noisy, t, c)
            loss    = tail_weighted_loss(pred, noise)
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
        sched.step()
        avg_loss = epoch_loss / len(loader)

        if avg_loss < best_loss:
            best_loss    = avg_loss
            best_weights = copy.deepcopy(model.state_dict())
            patience_ctr = 0
        else:
            patience_ctr += 1

        if epoch % 10 == 0:
            print(f"FT Epoch {epoch:>3d} | Loss: {avg_loss:.7f} | "
                  f"Best: {best_loss:.7f} | Patience: {patience_ctr}/{FT_PATIENCE}")

        if patience_ctr >= FT_PATIENCE:
            print(f"Early stopping at FT epoch {epoch}")
            break

    out_path = os.path.join(out_dir, "stage2_finetuned.pt")
    torch.save({
        "model_state_dict": best_weights,
        "best_loss"       : best_loss,
        "seq_len"         : seq_len,
        "n_assets"        : n_assets,
    }, out_path)
    print(f"Fine-tuned checkpoint saved: {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ohlcv_path",   required=True)
    ap.add_argument("--checkpoint",   required=True)
    ap.add_argument("--artefact_dir", default="artefacts")
    ap.add_argument("--out_dir",      default="checkpoints")
    ap.add_argument("--epochs",       type=int, default=FT_EPOCHS)
    ap.add_argument("--device",       default="auto")
    args = ap.parse_args()
    finetune(args.ohlcv_path, args.checkpoint, args.artefact_dir,
             args.out_dir, args.epochs, args.device)