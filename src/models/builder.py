from __future__ import annotations

import torch
from transformers import ViTConfig

from .attention import PrefilledAttention
from .preprocessor import LinearPreprocessor, compute_pca_matrix, compute_zca_matrix
from .specvit import MyViT
from src.utils import load_cov_stats

__all__ = ["get_model", "get_vit_config"]


def _get_freeze_suffix(freeze_epochs: int) -> str:
    """Get model name suffix for freeze status.
    
    Args:
        freeze_epochs: Freeze configuration
            > 0: temporary freeze for N epochs
            = -1: permanent freeze
            = 0: never frozen
    
    Returns:
        Suffix string: "perm" for permanent, number for temporary, "0" for never
    """
    return "perm" if freeze_epochs == -1 else str(freeze_epochs)


def _log_freeze_status(freeze_epochs: int) -> None:
    """Log preprocessor freeze status.
    
    Args:
        freeze_epochs: Freeze configuration
    """
    if freeze_epochs == -1:
        print("[builder] Preprocessor will be PERMANENTLY FROZEN (never trained)")
    elif freeze_epochs > 0:
        print(f"[builder] Preprocessor will be frozen for first {freeze_epochs} epochs")
    else:
        print("[builder] Preprocessor is trainable from start")


def get_model(config):
    """Build model with optional preprocessor (ZCA/PCA/Attention)"""
    vit_config = get_vit_config(config)
    warmup_cfg = config.get("warmup", {}) or {}
    loss_name = config.get("loss", {}).get("name", None)
    
    preproc_type = warmup_cfg.get("preprocessor", None)  # "zca", "pca", "attention", or "None"
    # Handle None, null, or string "None"
    if preproc_type is None or str(preproc_type).lower() in ("none", "null"):
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
    
    # Determine initial freeze state
    freeze_epochs = warmup_cfg.get("freeze_epochs", 0)
    initial_freeze = freeze_epochs != 0  # Frozen if != 0 (either temporary or permanent)
    fz_suffix = _get_freeze_suffix(freeze_epochs)
    
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
        
        # Model name
        rank_str = f"ZCA{r}" if r is not None else "ZCA"
        shrink_str = f"_s{int(shrinkage*10)}"
        model_name = f"{rank_str}_fz{fz_suffix}{shrink_str}_ViT"
        
        # Log
        rank_desc = f"low-rank ZCA with r={r}" if r else "full-rank ZCA"
        print(f"[builder] Created {rank_desc}, eps={eps}, shrinkage={shrinkage}")
        
    elif preproc_type == "pca":
        # PCA projection: V[:, :r].T @ x (low-rank or full-rank)
        r = warmup_cfg.get("r", None)
        P = compute_pca_matrix(eigvecs, r=r)
        preprocessor = LinearPreprocessor(P, freeze=initial_freeze)
        
        # Model name
        rank_str = f"PCA{r}" if r is not None else "PCA"
        model_name = f"{rank_str}_fz{fz_suffix}_ViT"
        
        # Log
        rank_desc = f"PCA with r={r}" if r else "full-rank PCA"
        print(f"[builder] Created {rank_desc} preprocessor")
        
    elif preproc_type == "attention":
        # Global attention with Q, K initialized from eigenvectors
        r = warmup_cfg.get("r", None)
        preprocessor = PrefilledAttention(input_dim=input_dim, eigvecs=eigvecs, r=r)
        
        # Model name
        rank_str = r if r else "Full"
        model_name = f"Attn{rank_str}_fz{fz_suffix}_ViT"
        
        # Log
        print(f"[builder] Created Attention preprocessor with r={r}")
        
    else:
        raise ValueError(f"Unknown preprocessor type: '{preproc_type}'")
    
    # Log freeze status
    _log_freeze_status(freeze_epochs)
    
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
    
    # Position encoding configuration (default: None - no position encoding)
    pos_encoding_type = m.get("pos_encoding_type", None)
    max_position_embeddings = m.get("max_position_embeddings", 512)
    rope_base = m.get("rope_base", 10000.0)

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
        pos_encoding_type=pos_encoding_type,
        max_position_embeddings=max_position_embeddings,
        rope_base=rope_base,
    )
