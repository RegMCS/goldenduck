"""
Test script for DDPM inference (Stage 1 & Stage 2)
Validates model outputs, inference pipelines, and end-to-end workflows.
Run with: python scripts/test_inference.py
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import logging
from pathlib import Path

# Setup paths
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from config.ddpm_config import TRAINING_CONFIG, STAGE1_CONFIG, STAGE2_CONFIG
from models.schedulers import DDPMScheduler
from models.denoiser import Conv1DUNet
from models.conditional_denoiser import ConditionalDenoiser
from services.inference import InferenceService

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def test_stage1_inference():
    """Test Stage 1 unconditional DDPM inference"""
    logger.info("=" * 60)
    logger.info("TEST: Stage 1 Unconditional Inference")
    logger.info("=" * 60)

    device = "cpu"
    
    # Create model and scheduler
    model = Conv1DUNet(
        num_assets=TRAINING_CONFIG["NUM_ASSETS"],
        base_channels=STAGE1_CONFIG["BASE_CHANNELS"],
        time_emb_dim=TRAINING_CONFIG["TIME_EMB_DIM"],
        depth=STAGE1_CONFIG["DEPTH"],
    ).to(device)
    
    scheduler = DDPMScheduler(T=STAGE1_CONFIG["T_STEPS"], device=device)
    
    # Create inference service
    inference = InferenceService(device=device)
    
    # Generate small batch for testing
    num_paths = 10
    logger.info(f"Generating {num_paths} Stage 1 paths...")
    
    paths = inference.generate_stage1(
        model=model,
        scheduler=scheduler,
        num_paths=num_paths,
        shape=(5, TRAINING_CONFIG["NUM_ASSETS"], TRAINING_CONFIG["WINDOW_LEN"]),
        seed=42,
    )
    
    # Validation
    assert paths.shape == (num_paths, TRAINING_CONFIG["NUM_ASSETS"], TRAINING_CONFIG["WINDOW_LEN"]), \
        f"Shape mismatch: {paths.shape}"
    assert not np.isnan(paths).any(), "Generated paths contain NaN"
    assert not np.isinf(paths).any(), "Generated paths contain Inf"
    
    logger.info(f"✓ Stage 1 output shape: {paths.shape}")
    logger.info(f"  Mean: {paths.mean():.6f}, Std: {paths.std():.6f}")
    logger.info(f"  Min: {paths.min():.6f}, Max: {paths.max():.6f}")

    return paths


def test_stage2_inference():
    """Test Stage 2 conditional DDPM inference with CFG"""
    logger.info("=" * 60)
    logger.info("TEST: Stage 2 Conditional Inference with CFG")
    logger.info("=" * 60)

    device = "cpu"
    
    # Create model and scheduler
    model = ConditionalDenoiser(
        seq_len=TRAINING_CONFIG["WINDOW_LEN"],
        in_ch=TRAINING_CONFIG["NUM_ASSETS"],
        cond_dim=STAGE2_CONFIG["COND_DIM"],
        time_emb_dim=TRAINING_CONFIG["TIME_EMB_DIM"],
        base_ch=STAGE2_CONFIG["BASE_CHANNELS"],
    ).to(device)
    
    scheduler = DDPMScheduler(T=STAGE2_CONFIG["T_STEPS"], device=device)
    
    # Create inference service
    inference = InferenceService(device=device)
    
    # Test conditioning vector: [realised_vol, drift, tail_index, momentum]
    conditioning = np.array([0.25, 0.05, 0.03, 0.15], dtype=np.float32)
    
    # Test different guidance scales
    guidance_scales = [1.0, 3.0, 5.0]
    
    for guidance_scale in guidance_scales:
        logger.info(f"\nTesting guidance_scale={guidance_scale}")
        
        num_paths = 5
        paths = inference.generate_stage2_cfg(
            model=model,
            scheduler=scheduler,
            conditioning=conditioning,
            num_paths=num_paths,
            guidance_scale=guidance_scale,
            shape=(5, TRAINING_CONFIG["NUM_ASSETS"], TRAINING_CONFIG["WINDOW_LEN"]),
            seed=42,
        )
        
        # Validation
        assert paths.shape == (num_paths, TRAINING_CONFIG["NUM_ASSETS"], TRAINING_CONFIG["WINDOW_LEN"]), \
            f"Shape mismatch: {paths.shape}"
        assert not np.isnan(paths).any(), "Generated paths contain NaN"
        assert not np.isinf(paths).any(), "Generated paths contain Inf"
        
        logger.info(f"  ✓ Output shape: {paths.shape}")
        logger.info(f"    Mean: {paths.mean():.6f}, Std: {paths.std():.6f}")


def test_unified_interface():
    """Test unified generate_paths() interface"""
    logger.info("=" * 60)
    logger.info("TEST: Unified generate_paths() Interface")
    logger.info("=" * 60)

    device = "cpu"
    inference = InferenceService(device=device)
    
    # Stage 1
    logger.info("\nStage 1 via unified interface:")
    model1 = Conv1DUNet(
        num_assets=TRAINING_CONFIG["NUM_ASSETS"],
        base_channels=STAGE1_CONFIG["BASE_CHANNELS"],
        time_emb_dim=TRAINING_CONFIG["TIME_EMB_DIM"],
        depth=STAGE1_CONFIG["DEPTH"],
    ).to(device)
    scheduler1 = DDPMScheduler(T=STAGE1_CONFIG["T_STEPS"], device=device)
    
    paths1 = inference.generate_paths(
        model=model1,
        scheduler=scheduler1,
        stage=1,
        num_paths=5,
        seed=42,
    )
    logger.info(f"  Shape: {paths1.shape}")
    assert paths1.shape[0] == 5, "Path count mismatch"
    
    # Stage 2
    logger.info("\nStage 2 via unified interface:")
    model2 = ConditionalDenoiser(
        seq_len=TRAINING_CONFIG["WINDOW_LEN"],
        in_ch=TRAINING_CONFIG["NUM_ASSETS"],
        cond_dim=STAGE2_CONFIG["COND_DIM"],
        time_emb_dim=TRAINING_CONFIG["TIME_EMB_DIM"],
        base_ch=STAGE2_CONFIG["BASE_CHANNELS"],
    ).to(device)
    scheduler2 = DDPMScheduler(T=STAGE2_CONFIG["T_STEPS"], device=device)
    
    conditioning = np.array([0.30, 0.02, 0.04, -0.10], dtype=np.float32)
    paths2 = inference.generate_paths(
        model=model2,
        scheduler=scheduler2,
        stage=2,
        num_paths=5,
        conditioning=conditioning,
        guidance_scale=4.0,
        seed=42,
    )
    logger.info(f"  Shape: {paths2.shape}")
    assert paths2.shape[0] == 5, "Path count mismatch"
    
    logger.info("✓ Unified interface tests passed")


def test_batch_generation():
    """Test batch generation (>1 batch needed)"""
    logger.info("=" * 60)
    logger.info("TEST: Batch Generation (multi-batch)")
    logger.info("=" * 60)

    device = "cpu"
    model = Conv1DUNet(
        num_assets=TRAINING_CONFIG["NUM_ASSETS"],
        base_channels=STAGE1_CONFIG["BASE_CHANNELS"],
        time_emb_dim=TRAINING_CONFIG["TIME_EMB_DIM"],
        depth=STAGE1_CONFIG["DEPTH"],
    ).to(device)
    scheduler = DDPMScheduler(T=STAGE1_CONFIG["T_STEPS"], device=device)
    inference = InferenceService(device=device)
    
    # Request more paths than batch size
    num_paths = 25
    batch_size = 10
    
    logger.info(f"Generating {num_paths} paths with batch_size={batch_size}")
    paths = inference.generate_stage1(
        model=model,
        scheduler=scheduler,
        num_paths=num_paths,
        shape=(batch_size, TRAINING_CONFIG["NUM_ASSETS"], TRAINING_CONFIG["WINDOW_LEN"]),
        seed=42,
    )
    
    assert paths.shape[0] == num_paths, f"Expected {num_paths} paths, got {paths.shape[0]}"
    logger.info(f"✓ Generated {paths.shape[0]} paths across multiple batches")


def test_reproducibility():
    """Test that seeding produces reproducible results"""
    logger.info("=" * 60)
    logger.info("TEST: Reproducibility with Fixed Seeds")
    logger.info("=" * 60)

    device = "cpu"
    model = Conv1DUNet(
        num_assets=TRAINING_CONFIG["NUM_ASSETS"],
        base_channels=STAGE1_CONFIG["BASE_CHANNELS"],
        time_emb_dim=TRAINING_CONFIG["TIME_EMB_DIM"],
        depth=STAGE1_CONFIG["DEPTH"],
    ).to(device)
    scheduler = DDPMScheduler(T=STAGE1_CONFIG["T_STEPS"], device=device)
    inference = InferenceService(device=device)
    
    # Generate twice with same seed
    paths1 = inference.generate_stage1(
        model=model,
        scheduler=scheduler,
        num_paths=5,
        seed=123,
    )
    
    paths2 = inference.generate_stage1(
        model=model,
        scheduler=scheduler,
        num_paths=5,
        seed=123,
    )
    
    # Check reproducibility
    diff = np.abs(paths1 - paths2).max()
    logger.info(f"Max difference between seeded runs: {diff:.10f}")
    assert diff < 1e-5, "Seeding did not produce reproducible results"
    logger.info("✓ Seeded generation is reproducible")


def run_all_tests():
    """Run all tests"""
    logger.info("\n" + "=" * 60)
    logger.info("DDPM INFERENCE TEST SUITE")
    logger.info("=" * 60)

    try:
        test_stage1_inference()
        logger.info("\n✓ Stage 1 test passed\n")
    except Exception as e:
        logger.error(f"✗ Stage 1 test failed: {e}")
        return False

    try:
        test_stage2_inference()
        logger.info("\n✓ Stage 2 test passed\n")
    except Exception as e:
        logger.error(f"✗ Stage 2 test failed: {e}")
        return False

    try:
        test_unified_interface()
        logger.info("\n✓ Unified interface test passed\n")
    except Exception as e:
        logger.error(f"✗ Unified interface test failed: {e}")
        return False

    try:
        test_batch_generation()
        logger.info("\n✓ Batch generation test passed\n")
    except Exception as e:
        logger.error(f"✗ Batch generation test failed: {e}")
        return False

    try:
        test_reproducibility()
        logger.info("\n✓ Reproducibility test passed\n")
    except Exception as e:
        logger.error(f"✗ Reproducibility test failed: {e}")
        return False

    logger.info("=" * 60)
    logger.info("ALL TESTS PASSED ✓")
    logger.info("=" * 60)
    return True


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
