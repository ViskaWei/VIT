from __future__ import annotations

import torch


__all__ = ["build_model_name"]


def build_model_name(
    config,
    model_prefix: str = "ViT",
) -> str:
    """Build standardized model name from config"""
    stride_used = getattr(config, "stride_size", None)
    stride_tag = int(stride_used) if (stride_used is not None and stride_used) else config.stride_ratio
    
    return (
        f"{model_prefix}_p{config.patch_size}_h{config.hidden_size}_l{config.num_hidden_layers}_"
        f"a{config.num_attention_heads}_s{stride_tag}_p{config.proj_fn}"
    )
