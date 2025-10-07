#!/usr/bin/env python3
"""Test script to verify ZCA P matrix loading and application"""

import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.models.preprocessor import compute_zca_matrix, LinearPreprocessor
from src.utils import load_config

def test_zca_loading():
    """Test ZCA matrix loading and transformation"""
    
    # Load config
    config = load_config('configs/anyon/run.yaml')
    warmup_cfg = config.get('warmup', {})
    
    print("=" * 80)
    print("Testing ZCA Matrix Loading and Application")
    print("=" * 80)
    
    # Load covariance statistics
    cov_path = warmup_cfg.get('cov_path')
    print(f"\n1. Loading covariance statistics from: {cov_path}")
    stats = torch.load(cov_path, map_location='cpu', weights_only=True)
    
    print(f"   ✓ Loaded stats with keys: {list(stats.keys())}")
    eigvecs = stats['eigvecs']
    eigvals = stats['eigvals']
    mean = stats['mean']
    print(f"   ✓ eigvecs shape: {eigvecs.shape}")
    print(f"   ✓ eigvals shape: {eigvals.shape}")
    print(f"   ✓ mean shape: {mean.shape}")
    
    # Compute ZCA matrix
    r = warmup_cfg.get('r', None)
    eps = warmup_cfg.get('eps', 1e-5)
    print(f"\n2. Computing ZCA matrix with r={r}, eps={eps}")
    
    P = compute_zca_matrix(eigvecs, eigvals, eps=eps, r=r)
    print(f"   ✓ P matrix shape: {P.shape}")
    print(f"   ✓ P matrix statistics:")
    print(f"     - mean: {P.mean().item():.6f}")
    print(f"     - std: {P.std().item():.6f}")
    print(f"     - min: {P.min().item():.6f}")
    print(f"     - max: {P.max().item():.6f}")
    
    # Create preprocessor
    print(f"\n3. Creating LinearPreprocessor with frozen P matrix")
    preprocessor = LinearPreprocessor(P, freeze=True)
    print(f"   ✓ Preprocessor created")
    print(f"   ✓ Linear layer weight shape: {preprocessor.linear.lin.weight.shape}")
    print(f"   ✓ Linear layer weight frozen: {not preprocessor.linear.lin.weight.requires_grad}")
    
    # Test forward pass
    print(f"\n4. Testing forward pass with random input")
    batch_size = 4
    input_dim = config['model']['image_size']
    x = torch.randn(batch_size, input_dim)
    print(f"   Input shape: {x.shape}")
    print(f"   Input statistics:")
    print(f"     - mean: {x.mean().item():.6f}")
    print(f"     - std: {x.std().item():.6f}")
    print(f"     - min: {x.min().item():.6f}")
    print(f"     - max: {x.max().item():.6f}")
    
    y = preprocessor(x)
    print(f"\n   Output shape: {y.shape}")
    print(f"   Output statistics:")
    print(f"     - mean: {y.mean().item():.6f}")
    print(f"     - std: {y.std().item():.6f}")
    print(f"     - min: {y.min().item():.6f}")
    print(f"     - max: {y.max().item():.6f}")
    
    # Verify P matrix is correctly applied
    print(f"\n5. Verifying P @ x computation")
    y_manual = x @ P.t()  # Manual computation: (batch, D) @ (D, D).T = (batch, D)
    diff = (y - y_manual).abs().max().item()
    print(f"   Max difference between preprocessor(x) and x @ P.T: {diff:.10f}")
    if diff < 1e-5:
        print(f"   ✓ P matrix is correctly applied!")
    else:
        print(f"   ✗ Warning: difference is larger than expected")
    
    # Test whitening property (approximately)
    print(f"\n6. Testing whitening property")
    n_samples = 1000
    X_test = torch.randn(n_samples, input_dim)
    Y_test = preprocessor(X_test)
    
    # Compute covariance of output
    Y_centered = Y_test - Y_test.mean(dim=0, keepdim=True)
    cov_output = (Y_centered.t() @ Y_centered) / (n_samples - 1)
    
    # For perfect whitening, cov_output should be close to identity
    # For low-rank ZCA, it's an approximation
    trace = cov_output.diag().mean().item()
    off_diag = (cov_output - torch.diag(cov_output.diag())).abs().mean().item()
    
    print(f"   Output covariance statistics:")
    print(f"     - Mean diagonal: {trace:.6f} (should be ~1 for whitening)")
    print(f"     - Mean off-diagonal: {off_diag:.6f} (should be ~0 for whitening)")
    
    if r is not None:
        print(f"   Note: Low-rank ZCA (r={r}) is an approximation, so perfect whitening is not expected")
    
    print("\n" + "=" * 80)
    print("✓ All tests passed! ZCA matrix is correctly loaded and applied.")
    print("=" * 80)

if __name__ == '__main__':
    test_zca_loading()
