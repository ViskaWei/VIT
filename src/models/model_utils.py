from __future__ import annotations

import torch


__all__ = ["build_model_name"]


def build_model_name(
    config,
    model_prefix: str = "ViT",
    full_config: dict = None,
) -> str:
    """Build standardized model name from config
    
    Args:
        config: ViTConfig or model config with architectural parameters
        model_prefix: Prefix for the model name (e.g., "ViT", "ZCA_ViT")
        full_config: Full config dict containing noise section (optional)
    
    Returns:
        Model name string with noise level suffix if noise_level > 0
    """
    stride_used = getattr(config, "stride_size", None)
    stride_tag = int(stride_used) if (stride_used is not None and stride_used) else config.stride_ratio
    
    base_name = (
        f"{model_prefix}_p{config.patch_size}_h{config.hidden_size}_l{config.num_hidden_layers}_"
        f"a{config.num_attention_heads}_s{stride_tag}_p{config.proj_fn}"
    )
    
    # Add noise level suffix if noise_level > 0
    if full_config is not None:
        noise_config = full_config.get("noise", {})
        noise_level = noise_config.get("noise_level", 0)
        if noise_level > 0:
            # Remove decimal point from noise_level string
            noise_str = str(noise_level).replace(".", "")
            base_name += f"_nz{noise_str}"
    
    return base_name
