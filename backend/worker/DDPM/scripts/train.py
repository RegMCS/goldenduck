
import os, copy, argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from worker.DDPM.scripts.model import ConditionalDenoiser
from worker.DDPM.utils import get_noise_schedule, q_sample, tail_weighted_loss

CFG_DROP_PROB = 0.10
EPOCHS        = 300
PATIENCE      = 20
BATCH_SIZE    = 32
LR            = 1e-4
WEIGHT_DECAY  = 1e-5
GRAD_CLIP     = 1.0


def load_artefacts(artefact_dir: str):
    windows_norm   = np.load(f"{artefact_dir}/windows_norm.npy")
    C_norm         = np.load(f"{artefact_dir}/C_norm.npy")
    sample_weights = np.load(f"{artefact_dir}/sample_weights.npy")
    return windows_norm, C_norm, sample_weights


def train(artefact_dir: str = "artefacts",
          out_dir: str      = "checkpoints",
          epochs: int       = EPOCHS,
          device_str: str   = "auto"):

    os.makedirs(out_dir, exist_ok=True)
    device = torch.device(
        "cuda" if (device_str == "auto" and torch.cuda.is_available())
        else device_str if device_str != "auto" else "cpu"
    )
    print(f"Device: {device}")

    windows_norm, C_norm, sample_weights = load_artefacts(artefact_dir)
    n_windows, n_assets, seq_len = windows_norm.shape
    print(f"Windows: {windows_norm.shape}  Conditioning: {C_norm.shape}")

    X = torch.tensor(windows_norm, dtype=torch.float32)
    C = torch.tensor(C_norm,       dtype=torch.float32)

    dataset = TensorDataset(X, C)
    sampler = WeightedRandomSampler(
        weights     = torch.tensor(sample_weights, dtype=torch.float64),
        num_samples = len(dataset),
        replacement = True
    )
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=sampler)

    T = 200
    betas, alphas, alpha_bar = get_noise_schedule(T=T, device=str(device))

    model = ConditionalDenoiser(seq_len=seq_len, in_ch=n_assets, cond_dim=4).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    opt   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best_loss    = float("inf")
    best_weights = None
    patience_ctr = 0
    loss_history = []

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0

        for x0, c in loader:
            x0, c = x0.to(device), c.to(device)
            if np.random.rand() < CFG_DROP_PROB:
                c = torch.zeros_like(c)
            noise   = torch.randn_like(x0)
            t       = torch.randint(0, T, (x0.shape[0],), device=device)
            x_noisy = q_sample(x0, t, noise, alpha_bar)
            pred    = model(x_noisy, t, c)
            loss    = tail_weighted_loss(pred, noise)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
            opt.step()
            epoch_loss += loss.item()

        sched.step()
        avg_loss = epoch_loss / len(loader)
        loss_history.append(avg_loss)

        if avg_loss < best_loss:
            best_loss    = avg_loss
            best_weights = copy.deepcopy(model.state_dict())
            patience_ctr = 0
        else:
            patience_ctr += 1

        if epoch % 20 == 0:
            print(f"Epoch {epoch:>4d} | Loss: {avg_loss:.7f} | "
                  f"Best: {best_loss:.7f} | Patience: {patience_ctr}/{PATIENCE}")

        if patience_ctr >= PATIENCE:
            print(f"Early stopping at epoch {epoch}")
            break

    model.load_state_dict(best_weights)
    torch.save({
        "model_state_dict": best_weights,
        "loss_history"    : loss_history,
        "best_loss"       : best_loss,
        "seq_len"         : seq_len,
        "n_assets"        : n_assets,
    }, f"{out_dir}/stage2_best.pt")
    np.save(f"{out_dir}/loss_history.npy", np.array(loss_history))
    print(f"Checkpoint saved: {out_dir}/stage2_best.pt")
    print(f"Best loss: {best_loss:.7f}")
    return model


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--artefact_dir", default="artefacts")
    ap.add_argument("--out_dir",      default="checkpoints")
    ap.add_argument("--epochs",       type=int, default=EPOCHS)
    ap.add_argument("--device",       default="auto")
    args = ap.parse_args()
    train(args.artefact_dir, args.out_dir, args.epochs, args.device)