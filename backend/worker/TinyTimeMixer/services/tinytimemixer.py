"""
Uses IBM Granite TinyTimeMixer (TTM) pretrained models hosted on Hugging Face:
- ibm-granite/granite-timeseries-ttm-r1
- ibm-granite/granite-timeseries-ttm-r2

Input convention (simple + explicit):
- past_values: torch.Tensor of shape [batch, context_length, channels]
- optional: past_observed_mask: same shape, 1 for observed, 0 for missing (if you have missing)

Output:
- forecast: torch.Tensor of shape [batch, prediction_length, channels]

Notes:
- For "zero-shot" inference, you typically just load the pretrained model and call predict().
- For fine-tuning, you should use TimeSeriesPreprocessor + Hugging Face Trainer (separate pipeline).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Optional, Tuple, Dict, Any

import torch

TTMVariant = Literal["r1", "r2"]

MODEL_IDS: Dict[str, str] = {
    "r1": "ibm-granite/granite-timeseries-ttm-r1",
    "r2": "ibm-granite/granite-timeseries-ttm-r2",
}


@dataclass
class ChannelScaler:
    """
    Per-channel standard scaler fitted on the provided context window.
    """
    mean_: torch.Tensor  # [batch, 1, channels] or [1, 1, channels]
    std_: torch.Tensor   # [batch, 1, channels] or [1, 1, channels]
    eps: float = 1e-6

    @classmethod
    def fit(cls, x: torch.Tensor, eps: float = 1e-6) -> "ChannelScaler":
        """
        Fit scaler on x of shape [B, T, C].
        Computes mean/std over T, separately per batch and channel.
        """
        if x.ndim != 3:
            raise ValueError(f"Expected x shape [B,T,C], got {tuple(x.shape)}")

        mean_ = x.mean(dim=1, keepdim=True)  # [B,1,C]
        var_ = x.var(dim=1, keepdim=True, unbiased=False)
        std_ = torch.sqrt(var_ + eps)
        return cls(mean_=mean_, std_=std_, eps=eps)

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean_) / (self.std_ + self.eps)

    def inverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        return x * (self.std_ + self.eps) + self.mean_


class TinyTimeMixer:
    """
    Wrapper for loading + running IBM Granite TinyTimeMixer (TTM) models.

    Uses tsfm_public.models.tinytimemixer.TinyTimeMixerForPrediction under the hood.
    """

    def __init__(
        self,
        variant: TTMVariant = "r2",
        device: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None,
        *,
        # Overrides you might want when adapting the checkpoint to your dataset:
        num_input_channels: Optional[int] = None,
        prediction_length: Optional[int] = None,
        decoder_mode: Optional[str] = None,
        prediction_channel_indices: Optional[list[int]] = None,
        exogenous_channel_indices: Optional[list[int]] = None,
    ) -> None:
        self.variant = variant
        self.model_id = MODEL_IDS[variant]

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.torch_dtype = torch_dtype

        self.model = self._load_model(
            model_id=self.model_id,
            torch_dtype=torch_dtype,
            num_input_channels=num_input_channels,
            prediction_length=prediction_length,
            decoder_mode=decoder_mode,
            prediction_channel_indices=prediction_channel_indices,
            exogenous_channel_indices=exogenous_channel_indices,
        ).to(self.device)

        self.model.eval()

    @staticmethod
    def _load_model(
        model_id: str,
        torch_dtype: Optional[torch.dtype],
        num_input_channels: Optional[int],
        prediction_length: Optional[int],
        decoder_mode: Optional[str],
        prediction_channel_indices: Optional[list[int]],
        exogenous_channel_indices: Optional[list[int]],
    ):
        """
        Load TinyTimeMixerForPrediction via TSFM.

        If you see import errors:
        - Ensure granite-tsfm is installed.
        - Pin transformers close to the model's config version (e.g., 4.37.x for R2 configs). :contentReference[oaicite:6]{index=6}
        """
        try:
            from tsfm_public.models.tinytimemixer import TinyTimeMixerForPrediction
        except Exception as e:
            raise ImportError(
                "Failed to import TinyTimeMixerForPrediction from tsfm_public. "
                "Install IBM Granite TSFM library (granite-tsfm / tsfm_public)."
            ) from e

        extra_kwargs: Dict[str, Any] = {}
        if num_input_channels is not None:
            extra_kwargs["num_input_channels"] = int(num_input_channels)
        if prediction_length is not None:
            extra_kwargs["prediction_length"] = int(prediction_length)
        if decoder_mode is not None:
            extra_kwargs["decoder_mode"] = decoder_mode
        if prediction_channel_indices is not None:
            extra_kwargs["prediction_channel_indices"] = prediction_channel_indices
        if exogenous_channel_indices is not None:
            extra_kwargs["exogenous_channel_indices"] = exogenous_channel_indices
        if torch_dtype is not None:
            extra_kwargs["torch_dtype"] = torch_dtype

        # IBM Granite docs show this from_pretrained pattern for fine-tuning as well. :contentReference[oaicite:7]{index=7}
        model = TinyTimeMixerForPrediction.from_pretrained(model_id, **extra_kwargs)
        return model

    @torch.inference_mode()
    def predict(
        self,
        past_values: torch.Tensor,
        *,
        past_observed_mask: Optional[torch.Tensor] = None,
        scale: bool = True,
        return_scaled: bool = False,
    ) -> Tuple[torch.Tensor, Optional[ChannelScaler]]:
        """
        Run inference.

        Args:
            past_values: [B, context_length, C]
            past_observed_mask: optional [B, context_length, C]
            scale: if True, fit per-channel scaler on past_values and feed scaled values
            return_scaled: if True, return the scaled forecast instead of inverse-transformed

        Returns:
            forecast: [B, prediction_length, C]
            scaler: ChannelScaler if scale=True else None
        """
        if past_values.ndim != 3:
            raise ValueError(f"past_values must be [B,T,C], got {tuple(past_values.shape)}")

        x = past_values.to(self.device)
        mask = past_observed_mask.to(self.device) if past_observed_mask is not None else None

        scaler: Optional[ChannelScaler] = None
        if scale:
            scaler = ChannelScaler.fit(x)
            x_in = scaler.transform(x)
        else:
            x_in = x

        # Try a few common argument names used in TSFM/Transformers-style forward signatures.
        # We keep it defensive so small upstream API changes don't break your project.
        outputs = None
        forward_errors = []

        for kwargs in (
            {"past_values": x_in, "past_observed_mask": mask},
            {"inputs": x_in, "past_observed_mask": mask},
            {"x": x_in, "mask": mask},
            {"past_values": x_in},
            {"inputs": x_in},
            {"x": x_in},
        ):
            try:
                # Drop None values so we don't pass mask=None if not supported
                clean_kwargs = {k: v for k, v in kwargs.items() if v is not None}
                outputs = self.model(**clean_kwargs)
                break
            except Exception as e:
                forward_errors.append((kwargs, repr(e)))

        if outputs is None:
            err_preview = "\n".join([f"kwargs={k} -> {e}" for k, e in forward_errors[:3]])
            raise RuntimeError(
                "TinyTimeMixer forward() failed with several common signatures.\n"
                "First errors:\n" + err_preview
            )

        # Normalize output extraction.
        # Depending on TSFM version, it may return:
        # - a ModelOutput-like object with .predictions / .prediction_outputs / .logits
        # - a tuple where first element is prediction tensor
        y_hat = None

        if hasattr(outputs, "predictions"):
            y_hat = outputs.predictions
        elif hasattr(outputs, "prediction_outputs"):
            y_hat = outputs.prediction_outputs
        elif hasattr(outputs, "logits"):
            y_hat = outputs.logits
        elif isinstance(outputs, (tuple, list)) and len(outputs) > 0 and torch.is_tensor(outputs[0]):
            y_hat = outputs[0]
        elif torch.is_tensor(outputs):
            y_hat = outputs

        if y_hat is None or not torch.is_tensor(y_hat):
            raise RuntimeError(
                "Could not extract forecast tensor from model outputs. "
                f"Got output type: {type(outputs)} with attrs: {dir(outputs)[:20]}"
            )

        # Ensure [B, pred_len, C] ordering
        # If upstream returns [B, C, pred_len], fix it.
        if y_hat.ndim == 3:
            b, a, c = y_hat.shape
            # Heuristic: if middle dim equals channels and last dim equals pred_len, it might be [B,C,H]
            # We detect channels by matching past_values C.
            past_c = past_values.shape[-1]
            if a == past_c and c != past_c:
                y_hat = y_hat.transpose(1, 2)  # [B,H,C]

        if return_scaled or (not scale):
            return y_hat.detach().cpu(), scaler

        if scaler is None:
            return y_hat.detach().cpu(), None

        y_out = scaler.inverse_transform(y_hat)
        return y_out.detach().cpu(), scaler


def load_ttm_r1(
    device: Optional[str] = None,
    torch_dtype: Optional[torch.dtype] = None,
    **kwargs,
) -> TinyTimeMixer:
    return TinyTimeMixer("r1", device=device, torch_dtype=torch_dtype, **kwargs)


def load_ttm_r2(
    device: Optional[str] = None,
    torch_dtype: Optional[torch.dtype] = None,
    **kwargs,
) -> TinyTimeMixer:
    return TinyTimeMixer("r2", device=device, torch_dtype=torch_dtype, **kwargs)
