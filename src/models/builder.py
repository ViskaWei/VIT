from __future__ import annotations

import torch
from transformers import ViTConfig

from .attention import PrefilledAttention
from .preprocessor import LinearPreprocessor, compute_pca_matrix, compute_zca_matrix
from .specvit import MyViT
from src.utils import load_cov_stats

__all__ = ["get_model", "get_vit_config"]


def get_model(config):
    """Build model with optional preprocessor (ZCA/PCA/Attention)"""
    vit_config = get_vit_config(config)
    warmup_cfg = config.get("warmup", {}) or {}
    loss_name = config.get("loss", {}).get("name", None)
    
    preproc_type = warmup_cfg.get("preprocessor", None)  # "zca", "pca", "attention"
    if preproc_type is None:
        # No preprocessor
        model = MyViT(vit_config, loss_name=loss_name, model_name="ViT")
        print(f"[builder] Created vanilla ViT model")
        return model
    
    # Load covariance statistics
    cov_path = warmup_cfg.get("cov_path", None)
    if cov_path is None:
        raise ValueError(f"preprocessor='{preproc_type}' requires 'cov_path' in warmup config")
    
    stats = load_cov_stats(cov_path)
    eigvecs = stats["eigvecs"]  # (input_dim, input_dim)
    input_dim = eigvecs.shape[0]
    
    if input_dim != vit_config.image_size:
        raise ValueError(
            f"Mismatch: eigvecs dimension {input_dim} != image_size {vit_config.image_size}"
        )
    
    # Determine initial freeze state (frozen if freeze_epochs > 0)
    freeze_epochs = warmup_cfg.get("freeze_epochs", 0)
    initial_freeze = freeze_epochs > 0
    
    preprocessor = None
    model_name = "ViT"
    
    if preproc_type == "zca":
        # ZCA whitening: P @ x where P.T @ cov @ P = I
        # Supports both full-rank (r=None) and low-rank (r>0) ZCA
        eigvals = stats["eigvals"]
        eps = warmup_cfg.get("eps", 1e-5)
        r = warmup_cfg.get("r", None)
        shrinkage = warmup_cfg.get("shrinkage", 0.0)
        
        P = compute_zca_matrix(eigvecs, eigvals, eps=eps, r=r, shrinkage=shrinkage)
        preprocessor = LinearPreprocessor(P, freeze=initial_freeze)
        
        if r is not None:
            model_name = f"ZCA{r}_fz{freeze_epochs}_s{int(shrinkage*10)}_ViT"
            print(f"[builder] Created low-rank ZCA preprocessor with r={r}, eps={eps}, shrinkage={shrinkage}")
        else:
            model_name = f"ZCA_fz{freeze_epochs}_s{int(shrinkage*10)}_ViT"
            print(f"[builder] Created full-rank ZCA preprocessor with eps={eps}, shrinkage={shrinkage}")
        
    elif preproc_type == "pca":
        # PCA projection: V[:, :r].T @ x (low-rank or full-rank)
        r = warmup_cfg.get("r", None)
        P = compute_pca_matrix(eigvecs, r=r)
        preprocessor = LinearPreprocessor(P, freeze=initial_freeze)
        if r is not None:
            model_name = f"PCA{r}_fz{freeze_epochs}_ViT"
            print(f"[builder] Created PCA preprocessor with r={r}")
        else:
            model_name = f"PCA_fz{freeze_epochs}_ViT"
            print(f"[builder] Created full-rank PCA preprocessor")
        
    elif preproc_type == "attention":
        # Global attention with Q, K initialized from eigenvectors
        r = warmup_cfg.get("r", None)
        preprocessor = PrefilledAttention(input_dim=input_dim, eigvecs=eigvecs, r=r)
        model_name = f"Attn{r if r else 'Full'}_fz{freeze_epochs}_ViT"
        print(f"[builder] Created Attention preprocessor with r={r}")
        
    else:
        raise ValueError(f"Unknown preprocessor type: '{preproc_type}'")
    
    # Log freeze status
    if freeze_epochs > 0:
        print(f"[builder] Preprocessor will be frozen for first {freeze_epochs} epochs")
    
    model = MyViT(
        vit_config,
        loss_name=loss_name,
        model_name=model_name,
        preprocessor=preprocessor,
    )
    
    print(f"[builder] Created {model._model_name} with {preproc_type} preprocessor")
    return model


def get_vit_config(config):
    """Build ViTConfig from config dict"""
    m = config["model"]
    d = config.get("data", {})
    num_labels = int(m.get("num_labels", 1) or 1)
    task = (m.get("task_type") or m.get("task") or "cls").lower()
    
    if task in ("reg", "regression"):
        p = d.get("param", None)
        if isinstance(p, str) and len(p) > 0:
            plist = [x.strip() for x in p.split(",") if x.strip()]
            if len(plist) >= 1:
                num_labels = len(plist)
        elif isinstance(p, (list, tuple)) and len(p) > 0:
            num_labels = len(p)
        m["num_labels"] = num_labels

    return ViTConfig(
        task_type=m["task_type"],
        image_size=m["image_size"],
        patch_size=m["patch_size"],
        num_channels=1,
        hidden_size=m["hidden_size"],
        num_hidden_layers=m["num_hidden_layers"],
        num_attention_heads=m["num_attention_heads"],
        intermediate_size=4 * m["hidden_size"],
        stride_ratio=m.get("stride_ratio", 1),
        stride_size=m.get("stride_size", None),
        proj_fn=m["proj_fn"],
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        is_encoder_decoder=False,
        use_mask_token=False,
        qkv_bias=True,
        num_labels=num_labels,
    )
