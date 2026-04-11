
import numpy as np
import torch
from worker.DDPM.scripts.model import ConditionalDenoiser
from worker.DDPM.utils import (get_noise_schedule, normalise_cond_vector,
                                REGIME_PRESETS)


class InferenceService:
    def __init__(self, checkpoint_path: str, artefact_dir: str,
                 device_str: str = "auto"):
        self.device = torch.device(
            "cuda" if (device_str == "auto" and torch.cuda.is_available())
            else device_str if device_str != "auto" else "cpu"
        )
        ckpt = torch.load(checkpoint_path, map_location=self.device)

        self.seq_len  = ckpt.get("seq_len",  1260)
        self.n_assets = ckpt.get("n_assets",    7)

        self.model = ConditionalDenoiser(
            seq_len  = self.seq_len,
            in_ch    = self.n_assets,
            cond_dim = 4,
        ).to(self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

        self.cond_norm_params = {
            "min": np.load(f"{artefact_dir}/cond_norm_min.npy"),
            "max": np.load(f"{artefact_dir}/cond_norm_max.npy"),
        }
        self.window_scales = np.load(f"{artefact_dir}/window_scales.npy")

        T = 200
        self.betas, self.alphas, self.alpha_bar = get_noise_schedule(
            T=T, device=str(self.device))
        self.T = T
        print(f"InferenceService loaded | seq_len={self.seq_len} "
              f"n_assets={self.n_assets} | device={self.device}")

    def _cond_tensor(self, raw_dict: dict) -> torch.Tensor:
        normed = normalise_cond_vector(raw_dict, self.cond_norm_params)
        return torch.tensor(normed, dtype=torch.float32,
                             device=self.device).unsqueeze(0)

    @torch.no_grad()
    def generate(self, regime: str = "crisis",
                 n_paths: int = 100,
                 guidance_scale: float = 3.0,
                 asset_idx: int = 0,
                 custom_cond: dict = None) -> np.ndarray:
        """
        Generate n_paths synthetic log-return paths.

        regime         : "calm" | "highvol" | "crisis" | "custom"
        n_paths        : number of paths to generate
        guidance_scale : CFG scale (higher = stronger conditioning)
        asset_idx      : which asset to extract (0 = SPY by default)
        custom_cond    : dict with realised_vol, drift, tail_index, momentum
                         (required when regime="custom")

        Returns: np.ndarray of shape (n_paths, seq_len)
        """
        if regime == "custom":
            if custom_cond is None:
                raise ValueError("custom_cond dict required when regime='custom'")
            raw_cond = custom_cond
        else:
            raw_cond = REGIME_PRESETS[regime]

        c_cond = self._cond_tensor(raw_cond).expand(n_paths, -1)
        c_null = torch.zeros_like(c_cond)

        x = torch.randn(n_paths, self.n_assets, self.seq_len,
                        device=self.device)

        for t_idx in reversed(range(self.T)):
            t_batch = torch.full((n_paths,), t_idx, dtype=torch.long,
                                  device=self.device)
            beta    = self.betas[t_idx]
            alpha   = self.alphas[t_idx]
            ab      = self.alpha_bar[t_idx]

            noise_cond = self.model(x, t_batch, c_cond)
            noise_null = self.model(x, t_batch, c_null)
            noise_pred = noise_null + guidance_scale * (noise_cond - noise_null)

            x0_pred = (x - torch.sqrt(1 - ab) * noise_pred) / torch.sqrt(ab)
            x0_pred = torch.clamp(x0_pred, -5.0, 5.0)

            if t_idx > 0:
                ab_prev = self.alpha_bar[t_idx - 1]
                coef1   = torch.sqrt(ab_prev) * beta / (1 - ab)
                coef2   = torch.sqrt(alpha) * (1 - ab_prev) / (1 - ab)
                mu      = coef1 * x0_pred + coef2 * x
                sigma   = torch.sqrt(beta * (1 - ab_prev) / (1 - ab))
                x       = mu + sigma * torch.randn_like(x)
            else:
                x = x0_pred

        paths_norm   = x[:, asset_idx, :].cpu().numpy()
        median_scale = float(np.median(self.window_scales[:, asset_idx]))
        return paths_norm * median_scale

    @torch.no_grad()
    def generate_all_regimes(self, n_paths: int = 100,
                              guidance_scale: float = 3.0,
                              asset_idx: int = 0) -> dict:
        """
        Generate paths for all 3 regime presets.
        Returns: {"calm": (N,L), "highvol": (N,L), "crisis": (N,L)}
        """
        results = {}
        for regime in ["calm", "highvol", "crisis"]:
            print(f"Generating {regime} paths ...")
            results[regime] = self.generate(
                regime         = regime,
                n_paths        = n_paths,
                guidance_scale = guidance_scale,
                asset_idx      = asset_idx,
            )
        return results