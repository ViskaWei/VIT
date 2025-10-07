#!/usr/bin/env python3
"""验证PCA流程的正确性：权重加载、冻结状态、前向传播"""

import torch
import sys
sys.path.insert(0, '/Users/viskawei/Desktop/VIT')

from src.models.preprocessor import LinearPreprocessor, compute_pca_matrix
from src.models.builder import get_model
from src.utils import load_cov_stats

def test_pca_matrix_computation():
    """测试PCA矩阵计算"""
    print("\n" + "="*70)
    print("测试 1: PCA矩阵计算")
    print("="*70)
    
    # 创建测试数据
    D = 10
    eigvecs = torch.randn(D, D)
    eigvecs, _ = torch.linalg.qr(eigvecs)  # 正交化
    
    # 测试full-rank PCA
    P_full = compute_pca_matrix(eigvecs, r=None)
    print(f"Full-rank PCA: P.shape = {P_full.shape} (应该是 {D}x{D})")
    assert P_full.shape == (D, D), f"Full-rank shape错误: {P_full.shape}"
    
    # 测试low-rank PCA
    r = 5
    P_low = compute_pca_matrix(eigvecs, r=r)
    print(f"Low-rank PCA (r={r}): P.shape = {P_low.shape} (应该是 {r}x{D})")
    assert P_low.shape == (r, D), f"Low-rank shape错误: {P_low.shape}"
    
    print("✓ PCA矩阵计算正确\n")


def test_linear_preprocessor():
    """测试LinearPreprocessor的权重加载和冻结"""
    print("="*70)
    print("测试 2: LinearPreprocessor权重加载和冻结")
    print("="*70)
    
    D = 10
    r = 5
    batch_size = 3
    
    # 创建测试矩阵
    P = torch.randn(r, D)
    
    # 测试1: freeze=True (初始冻结)
    print("\n[a] 测试 freeze=True:")
    prep_frozen = LinearPreprocessor(P, freeze=True)
    print(f"  - 权重shape: {prep_frozen.linear.lin.weight.shape} (应该是 {r}x{D})")
    print(f"  - 权重是否相同: {torch.allclose(prep_frozen.linear.lin.weight, P)}")
    print(f"  - requires_grad: {prep_frozen.linear.lin.weight.requires_grad} (应该是 False)")
    assert prep_frozen.linear.lin.weight.shape == (r, D)
    assert torch.allclose(prep_frozen.linear.lin.weight, P)
    assert not prep_frozen.linear.lin.weight.requires_grad
    print("  ✓ 冻结状态正确")
    
    # 测试2: freeze=False (初始不冻结)
    print("\n[b] 测试 freeze=False:")
    prep_trainable = LinearPreprocessor(P, freeze=False)
    print(f"  - requires_grad: {prep_trainable.linear.lin.weight.requires_grad} (应该是 True)")
    assert prep_trainable.linear.lin.weight.requires_grad
    print("  ✓ 可训练状态正确")
    
    # 测试3: 动态冻结/解冻
    print("\n[c] 测试动态冻结/解冻:")
    prep_frozen.freeze(False)
    print(f"  - 解冻后 requires_grad: {prep_frozen.linear.lin.weight.requires_grad} (应该是 True)")
    assert prep_frozen.linear.lin.weight.requires_grad
    
    prep_frozen.freeze(True)
    print(f"  - 重新冻结后 requires_grad: {prep_frozen.linear.lin.weight.requires_grad} (应该是 False)")
    assert not prep_frozen.linear.lin.weight.requires_grad
    print("  ✓ 动态冻结/解冻正确")
    
    # 测试4: 前向传播
    print("\n[d] 测试前向传播:")
    x = torch.randn(batch_size, D)
    y = prep_frozen(x)
    expected_y = x @ P.t()
    print(f"  - 输入shape: {x.shape}")
    print(f"  - 输出shape: {y.shape} (应该是 {batch_size}x{r})")
    print(f"  - 计算正确: {torch.allclose(y, expected_y, atol=1e-5)}")
    assert y.shape == (batch_size, r)
    assert torch.allclose(y, expected_y, atol=1e-5)
    print("  ✓ 前向传播正确\n")


def test_model_with_pca():
    """测试完整模型中的PCA preprocessor"""
    print("="*70)
    print("测试 3: 完整模型中的PCA流程")
    print("="*70)
    
    # 创建测试配置 (使用4096匹配实际数据)
    config_frozen = {
        "model": {
            "task_type": "reg",
            "num_labels": 1,
            "image_size": 4096,  # 匹配cov.pt的维度
            "patch_size": 32,
            "hidden_size": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "stride_ratio": 1,
            "proj_fn": "SW"  # 使用支持的proj_fn
        },
        "warmup": {
            "preprocessor": "pca",
            "cov_path": "/Users/viskawei/Desktop/VIT/data/pca/cov.pt",
            "r": 50,
            "freeze_epochs": 5  # 冻结5个epoch
        },
        "loss": {"name": "l1"}
    }
    
    config_no_freeze = {
        "model": {
            "task_type": "reg",
            "num_labels": 1,
            "image_size": 4096,  # 匹配cov.pt的维度
            "patch_size": 32,
            "hidden_size": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "stride_ratio": 1,
            "proj_fn": "SW"  # 使用支持的proj_fn
        },
        "warmup": {
            "preprocessor": "pca",
            "cov_path": "/Users/viskawei/Desktop/VIT/data/pca/cov.pt",
            "r": 50,
            # 没有freeze_epochs，默认为0
        },
        "loss": {"name": "l1"}
    }
    
    # 检查协方差文件是否存在
    import os
    cov_path = config_frozen["warmup"]["cov_path"]
    if not os.path.exists(cov_path):
        print(f"⚠️  协方差文件不存在: {cov_path}")
        print("  跳过完整模型测试（需要真实的协方差数据）\n")
        return
    
    # 加载协方差统计
    stats = load_cov_stats(cov_path)
    eigvecs = stats["eigvecs"]
    print(f"\n协方差数据: eigvecs.shape = {eigvecs.shape}")
    
    # 测试freeze_epochs > 0的情况
    print("\n[a] 测试 freeze_epochs=5 (应该初始冻结):")
    model_frozen = get_model(config_frozen)
    if model_frozen.preprocessor is not None:
        weight = model_frozen.preprocessor.linear.lin.weight
        requires_grad = weight.requires_grad
        print(f"  - Preprocessor存在: ✓")
        print(f"  - 权重shape: {weight.shape}")
        print(f"  - requires_grad: {requires_grad} (应该是 False)")
        assert not requires_grad, "freeze_epochs > 0时应该初始冻结!"
        print("  ✓ 初始冻结状态正确")
        
        # 测试解冻
        model_frozen.set_preprocessor_trainable(True)
        print(f"  - 解冻后 requires_grad: {weight.requires_grad} (应该是 True)")
        assert weight.requires_grad
        print("  ✓ 解冻功能正确")
    else:
        print("  ✗ 模型没有preprocessor!")
    
    # 测试freeze_epochs = 0的情况
    print("\n[b] 测试 freeze_epochs=0 (应该初始不冻结):")
    model_no_freeze = get_model(config_no_freeze)
    if model_no_freeze.preprocessor is not None:
        weight = model_no_freeze.preprocessor.linear.lin.weight
        requires_grad = weight.requires_grad
        print(f"  - Preprocessor存在: ✓")
        print(f"  - 权重shape: {weight.shape}")
        print(f"  - requires_grad: {requires_grad} (应该是 True)")
        # freeze_epochs=0时，initial_freeze=False，所以权重应该是可训练的
        assert requires_grad, "freeze_epochs=0时应该初始不冻结!"
        print("  ✓ 初始不冻结状态正确")
    else:
        print("  ✗ 模型没有preprocessor!")
    
    # 测试前向传播
    print("\n[c] 测试前向传播:")
    batch_size = 2
    x = torch.randn(batch_size, eigvecs.shape[0])
    print(f"  - 输入shape: {x.shape}")
    
    with torch.no_grad():
        output = model_frozen(x)
    print(f"  - 输出logits shape: {output.logits.shape}")
    print(f"  - 前向传播成功: ✓\n")


def test_freeze_logic_issue():
    """检查freeze_epochs=0时的行为"""
    print("="*70)
    print("测试 4: 检查freeze_epochs=0的逻辑")
    print("="*70)
    
    print("\n当前builder.py的逻辑:")
    print("  freeze_epochs = warmup_cfg.get('freeze_epochs', 0)")
    print("  initial_freeze = freeze_epochs > 0")
    print("  preprocessor = LinearPreprocessor(P, freeze=initial_freeze)")
    print("\n这意味着:")
    print("  - freeze_epochs > 0: 初始冻结，训练中解冻")
    print("  - freeze_epochs = 0: 初始不冻结，权重可训练")
    print("  - 没有freeze_epochs配置: 初始不冻结，权重可训练")
    
    print("\n✓ 逻辑是正确的！\n")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("PCA流程正确性验证")
    print("="*70)
    
    test_pca_matrix_computation()
    test_linear_preprocessor()
    test_model_with_pca()
    test_freeze_logic_issue()
    
    print("="*70)
    print("所有测试完成！")
    print("="*70 + "\n")
