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


def _build_preprocessor(preproc_type: str, warmup_cfg: dict, stats: dict, input_dim: int, initial_freeze: bool):
    """Build preprocessor and return (preprocessor, output_dim, model_name_prefix, description)
    
    Args:
        preproc_type: Type of preprocessor ("zca", "pca", "attention")
        warmup_cfg: Warmup configuration dict
        stats: Statistics dict with eigvecs, eigvals
        input_dim: Input dimension
        initial_freeze: Whether to freeze initially
    
    Returns:
        Tuple of (preprocessor, output_dim, model_name_prefix, description)
    """
    eigvecs = stats["eigvecs"]
    r = warmup_cfg.get("r", None)
    fz_suffix = _get_freeze_suffix(warmup_cfg.get("freeze_epochs", 0))
    
    if preproc_type == "zca":
        eigvals = stats["eigvals"]
        eps = warmup_cfg.get("eps", 1e-5)
        shrinkage = warmup_cfg.get("shrinkage", 0.0)
        
        P = compute_zca_matrix(eigvecs, eigvals, eps=eps, r=r, shrinkage=shrinkage)
        preprocessor = LinearPreprocessor(P, freeze=initial_freeze)
        output_dim = P.shape[0]  # Output dimension from ZCA matrix
        
        rank_str = f"ZCA{r}" if r is not None else "ZCA"
        shrink_str = f"_s{int(shrinkage*10)}"
        name_prefix = f"{rank_str}_fz{fz_suffix}{shrink_str}"
        desc = f"{'low-rank' if r else 'full-rank'} ZCA, eps={eps}, shrinkage={shrinkage}"
        
    elif preproc_type == "pca":
        P = compute_pca_matrix(eigvecs, r=r)
        preprocessor = LinearPreprocessor(P, freeze=initial_freeze)
        output_dim = P.shape[0]  # Output dimension from PCA matrix
        
        rank_str = f"PCA{r}" if r is not None else "PCA"
        name_prefix = f"{rank_str}_fz{fz_suffix}"
        desc = f"PCA with r={r}" if r else "full-rank PCA"
        
    elif preproc_type == "attention":
        preprocessor = PrefilledAttention(input_dim=input_dim, eigvecs=eigvecs, r=r)
        output_dim = r if r is not None else input_dim
        
        rank_str = r if r else "Full"
        name_prefix = f"Attn{rank_str}_fz{fz_suffix}"
        desc = f"Attention preprocessor with r={r}"
        
    else:
        raise ValueError(f"Unknown preprocessor type: '{preproc_type}'")
    
    return preprocessor, output_dim, name_prefix, desc


def get_model(config):
    """Build model with optional preprocessor (ZCA/PCA/Attention)
    
    Automatically adjusts image_size to match preprocessor output dimension.
    """
    warmup_cfg = config.get("warmup", {}) or {}
    loss_name = config.get("loss", {}).get("name", None)
    preproc_type = warmup_cfg.get("preprocessor", None)
    
    # Handle None, null, or string "None"
    if preproc_type is None or str(preproc_type).lower() in ("none", "null"):
        vit_config = get_vit_config(config)
        model = MyViT(vit_config, loss_name=loss_name, model_name="ViT")
        print(f"[builder] Created vanilla ViT model")
        return model
    
    # Load covariance statistics
    cov_path = warmup_cfg.get("cov_path", None)
    if cov_path is None:
        raise ValueError(f"preprocessor='{preproc_type}' requires 'cov_path' in warmup config")
    
    stats = load_cov_stats(cov_path)
    eigvecs = stats["eigvecs"]
    input_dim = eigvecs.shape[0]
    original_image_size = config["model"]["image_size"]
    
    if input_dim != original_image_size:
        raise ValueError(
            f"Mismatch: eigvecs dimension {input_dim} != image_size {original_image_size}"
        )
    
    # Build preprocessor and get output dimension
    freeze_epochs = warmup_cfg.get("freeze_epochs", 0)
    initial_freeze = freeze_epochs != 0
    
    preprocessor, output_dim, name_prefix, desc = _build_preprocessor(
        preproc_type, warmup_cfg, stats, input_dim, initial_freeze
    )
    
    # Auto-adjust image_size to match preprocessor output
    if output_dim != original_image_size:
        print(f"[builder] Auto-adjusting image_size: {original_image_size} → {output_dim}")
        config["model"]["image_size"] = output_dim
    
    # Build ViT config with adjusted image_size
    vit_config = get_vit_config(config)
    
    # Log preprocessor creation
    print(f"[builder] Created {desc} preprocessor")
    _log_freeze_status(freeze_epochs)
    
    # Build model
    model = MyViT(
        vit_config,
        loss_name=loss_name,
        model_name=f"{name_prefix}_ViT",
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
