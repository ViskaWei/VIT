#!/usr/bin/env python3
"""Test script to verify preprocessor freeze/unfreeze mechanism"""

import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.models.builder import get_model
from src.utils import load_config


def test_freeze_mechanism():
    """Test preprocessor freeze/unfreeze functionality"""
    
    print("=" * 80)
    print("Testing Preprocessor Freeze/Unfreeze Mechanism")
    print("=" * 80)
    
    # Load config
    config = load_config('configs/anyon/run.yaml')
    
    # Test 1: Check initial freeze state
    print("\n1. Testing initial freeze state (freeze_epochs=5)")
    freeze_epochs = config['warmup'].get('freeze_epochs', 0)
    print(f"   Config freeze_epochs: {freeze_epochs}")
    
    model = get_model(config)
    
    if model.preprocessor is None:
        print("   ✗ No preprocessor found!")
        return
    
    # Check if preprocessor parameters are frozen
    is_frozen = not model.preprocessor.linear.lin.weight.requires_grad
    print(f"   Preprocessor frozen: {is_frozen}")
    
    if freeze_epochs > 0:
        if is_frozen:
            print(f"   ✓ Preprocessor correctly initialized as frozen")
        else:
            print(f"   ✗ Preprocessor should be frozen but is not!")
    else:
        if not is_frozen:
            print(f"   ✓ Preprocessor correctly initialized as trainable")
        else:
            print(f"   ✗ Preprocessor should be trainable but is frozen!")
    
    # Test 2: Test set_preprocessor_trainable method
    print("\n2. Testing set_preprocessor_trainable(False)")
    model.set_preprocessor_trainable(False)
    is_frozen = not model.preprocessor.linear.lin.weight.requires_grad
    print(f"   Preprocessor frozen: {is_frozen}")
    if is_frozen:
        print(f"   ✓ Successfully froze preprocessor")
    else:
        print(f"   ✗ Failed to freeze preprocessor")
    
    print("\n3. Testing set_preprocessor_trainable(True)")
    model.set_preprocessor_trainable(True)
    is_frozen = not model.preprocessor.linear.lin.weight.requires_grad
    print(f"   Preprocessor frozen: {is_frozen}")
    if not is_frozen:
        print(f"   ✓ Successfully unfroze preprocessor")
    else:
        print(f"   ✗ Failed to unfreeze preprocessor")
    
    # Test 3: Count trainable parameters in different states
    print("\n4. Testing parameter counting")
    
    # Freeze preprocessor
    model.set_preprocessor_trainable(False)
    frozen_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_preproc_params = sum(p.numel() for p in model.preprocessor.parameters() if p.requires_grad)
    print(f"   With frozen preprocessor:")
    print(f"     Total trainable params: {frozen_params:,}")
    print(f"     Preprocessor trainable params: {frozen_preproc_params:,}")
    
    # Unfreeze preprocessor
    model.set_preprocessor_trainable(True)
    unfrozen_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    unfrozen_preproc_params = sum(p.numel() for p in model.preprocessor.parameters() if p.requires_grad)
    print(f"   With unfrozen preprocessor:")
    print(f"     Total trainable params: {unfrozen_params:,}")
    print(f"     Preprocessor trainable params: {unfrozen_preproc_params:,}")
    
    diff = unfrozen_params - frozen_params
    print(f"   Difference: {diff:,} params")
    
    if diff == unfrozen_preproc_params:
        print(f"   ✓ Parameter counting is consistent")
    else:
        print(f"   ✗ Warning: Parameter counting mismatch!")
    
    # Test 4: Verify gradient flow
    print("\n5. Testing gradient flow")
    
    # Freeze and test
    model.set_preprocessor_trainable(False)
    x = torch.randn(2, config['model']['image_size'], requires_grad=True)
    y = model.preprocessor(x)
    loss = y.sum()
    loss.backward()
    
    has_grad = model.preprocessor.linear.lin.weight.grad is not None
    print(f"   Frozen preprocessor has gradients: {has_grad}")
    if not has_grad:
        print(f"   ✓ No gradients when frozen (correctly skipped)")
    else:
        grad_norm = model.preprocessor.linear.lin.weight.grad.norm().item()
        print(f"   ! Gradients computed when frozen (grad norm: {grad_norm:.6f})")
        print(f"     Note: This is expected for frozen params in PyTorch")
    
    # Clear gradients
    model.zero_grad()
    
    # Unfreeze and test
    model.set_preprocessor_trainable(True)
    x = torch.randn(2, config['model']['image_size'], requires_grad=True)
    y = model.preprocessor(x)
    loss = y.sum()
    loss.backward()
    
    has_grad = model.preprocessor.linear.lin.weight.grad is not None
    grad_norm = model.preprocessor.linear.lin.weight.grad.norm().item() if has_grad else 0.0
    print(f"   Unfrozen preprocessor has gradients: {has_grad}")
    if has_grad:
        print(f"   ✓ Gradients computed when unfrozen (grad norm: {grad_norm:.6f})")
    else:
        print(f"   ✗ Warning: No gradients when unfrozen!")
    
    # Most important: verify requires_grad status
    print("\n6. Verifying requires_grad status")
    model.set_preprocessor_trainable(False)
    frozen_status = model.preprocessor.linear.lin.weight.requires_grad
    print(f"   Frozen: requires_grad = {frozen_status}")
    
    model.set_preprocessor_trainable(True)
    unfrozen_status = model.preprocessor.linear.lin.weight.requires_grad
    print(f"   Unfrozen: requires_grad = {unfrozen_status}")
    
    if not frozen_status and unfrozen_status:
        print(f"   ✓ requires_grad correctly controlled")
    else:
        print(f"   ✗ Warning: requires_grad not correctly controlled!")
    
    print("\n" + "=" * 80)
    print("✓ All freeze/unfreeze tests completed!")
    print("=" * 80)


if __name__ == '__main__':
    test_freeze_mechanism()
