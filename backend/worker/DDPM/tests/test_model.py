import pytest
import torch
from worker.DDPM.scripts.model import ConditionalDenoiser


@pytest.fixture
def model():
    return ConditionalDenoiser(seq_len=126, in_ch=7, cond_dim=4, base_ch=16)


def test_output_shape(model):
    x   = torch.randn(4, 7, 126)
    t   = torch.randint(0, 200, (4,))
    c   = torch.randn(4, 4)
    out = model(x, t, c)
    assert out.shape == (4, 7, 126), f"Expected (4,7,126) got {out.shape}"


def test_unconditional_zeros(model):
    x   = torch.randn(2, 7, 126)
    t   = torch.randint(0, 200, (2,))
    c   = torch.zeros(2, 4)
    out = model(x, t, c)
    assert out.shape == (2, 7, 126)


def test_conditioning_changes_output(model):
    x    = torch.randn(2, 7, 126)
    t    = torch.zeros(2, dtype=torch.long)
    c1   = torch.zeros(2, 4)
    c2   = torch.ones(2, 4)
    out1 = model(x, t, c1)
    out2 = model(x, t, c2)
    assert not torch.allclose(out1, out2), \
        "Conditioning vector has no effect on output"


def test_no_nan_output(model):
    x   = torch.randn(4, 7, 126)
    t   = torch.randint(0, 200, (4,))
    c   = torch.randn(4, 4)
    out = model(x, t, c)
    assert not torch.isnan(out).any(), "NaN detected in model output"