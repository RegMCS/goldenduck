import pytest
import numpy as np


@pytest.mark.integration
def test_generate_shape(inference_svc):
    paths = inference_svc.generate(regime="crisis", n_paths=5)
    assert paths.shape[0] == 5
    assert paths.shape[1] == inference_svc.seq_len


@pytest.mark.integration
def test_regime_outputs_differ(inference_svc):
    calm   = inference_svc.generate(regime="calm",    n_paths=10)
    crisis = inference_svc.generate(regime="crisis",  n_paths=10)
    calm_vol   = np.std(calm)   * np.sqrt(252)
    crisis_vol = np.std(crisis) * np.sqrt(252)
    assert crisis_vol > calm_vol, \
        f"Crisis vol ({crisis_vol:.3f}) should exceed calm vol ({calm_vol:.3f})"


@pytest.mark.integration
def test_no_explosive_paths(inference_svc):
    """Annualised vol should not exceed 5.0 (500%) — explosive path guard."""
    paths = inference_svc.generate(regime="crisis", n_paths=20)
    ann_vol = np.std(paths) * np.sqrt(252)
    assert ann_vol < 5.0, f"Explosive paths detected: annualised vol = {ann_vol:.3f}"