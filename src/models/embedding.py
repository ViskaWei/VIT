from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from .layers import complete_with_orthogonal, load_basis_matrix
from .tokenization import Conv1DPatchTokenizer, SlidingWindowTokenizer


__all__ = [
    "SpectraEmbeddings",
    "apply_patch_embed_pca",
]


class SpectraEmbeddings(nn.Module):
    """Patch + positional embeddings tailored for 1D spectral inputs."""

    def __init__(self, config: Any) -> None:
        super().__init__()
        stride_size = getattr(config, "stride_size", None)
        if config.proj_fn == "SW":
            self.patch_embeddings = SlidingWindowTokenizer(
                input_length=config.image_size,
                patch_size=config.patch_size,
                hidden_size=config.hidden_size,
                stride=stride_size if stride_size and stride_size > 0 else int(config.stride_ratio * config.patch_size),
            )
        elif config.proj_fn in ("C1D", "CNN"):
            self.patch_embeddings = Conv1DPatchTokenizer(
                input_length=config.image_size,
                patch_size=config.patch_size,
                hidden_size=config.hidden_size,
                stride=stride_size if stride_size and stride_size > 0 else int(config.stride_ratio * config.patch_size),
            )
        else:
            raise ValueError(f"Unsupported proj_fn '{config.proj_fn}' for SpectraEmbeddings")

        self.num_patches = self.patch_embeddings.num_patches
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.hidden_size))
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.position_embeddings = nn.Parameter(
            torch.randn(1, self.patch_embeddings.num_patches + 1, config.hidden_size)
        )

    def forward(
        self,
        x: torch.Tensor,
        bool_masked_pos: torch.BoolTensor | None = None,
        interpolate_pos_encoding: bool = False,
    ) -> torch.Tensor:
        tokens = self.patch_embeddings(x)
        batch_size, _, _ = tokens.size()
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat((cls_tokens, tokens), dim=1)
        tokens = tokens + self.position_embeddings
        return self.dropout(tokens)

    def set_patch_proj_trainable(self, trainable: bool = True) -> None:
        if hasattr(self.patch_embeddings, "projection"):
            for param in self.patch_embeddings.projection.parameters():
                param.requires_grad = trainable


def apply_patch_embed_pca(model: nn.Module, embed_cfg: dict[str, Any] | None) -> None:
    if embed_cfg is None:
        return
    try:
        emb = model.vit.embeddings.patch_embeddings
    except Exception:
        return
    proj = getattr(emb, "projection", None)
    if not isinstance(proj, (nn.Linear, nn.Conv1d)):
        return

    pth = embed_cfg.get("embed_pca_path", "pca_patch.pt")
    patch_dim = int(getattr(emb, "patch_size", proj.in_features if isinstance(proj, nn.Linear) else proj.kernel_size[0]))
    hidden = int(
        getattr(model.config, "hidden_size", getattr(proj, "out_features", getattr(proj, "out_channels", 0)))
    )

    basis_key = str(embed_cfg.get("UV", "V")) if embed_cfg is not None else "V"
    V = load_basis_matrix(pth, patch_dim, device=proj.weight.device, dtype=proj.weight.dtype, basis_key=basis_key)
    if V is None:
        return
    r = min(hidden, V.shape[1])
    V = V[:, :r].contiguous()
    if r < hidden:
        V = complete_with_orthogonal(V, out_dim=hidden)

    use_mean = bool(embed_cfg.get("use_pca_mean", False))
    mean_vec = None
    if use_mean:
        try:
            stats = torch.load(pth, map_location="cpu")
            if isinstance(stats, dict):
                mean_val = stats.get("mean", None)
                if isinstance(mean_val, torch.Tensor) and mean_val.dim() == 1 and int(mean_val.numel()) == patch_dim:
                    mean_vec = mean_val.to(device=V.device, dtype=V.dtype)
        except Exception:
            mean_vec = None

    with torch.no_grad():
        if isinstance(proj, nn.Linear):
            proj.weight.data.copy_(V.t())
            if proj.bias is not None:
                if mean_vec is not None:
                    proj.bias.data.copy_((-mean_vec @ V).to(proj.bias.dtype))
                else:
                    proj.bias.zero_()
        else:
            oc, ic, ksz = proj.weight.shape
            if ic != 1:
                print(f"[embed-warmup] Skip Conv1d PCA init: in_channels={ic} unsupported (expect 1)")
                return
            if ksz != patch_dim:
                print(
                    f"[embed-warmup] Warning: Conv1d kernel_size={ksz} != patch_dim={patch_dim}; proceeding with min size copy"
                )
            use_k = min(ksz, patch_dim)
            proj.weight.data.zero_()
            proj.weight.data[:hidden, 0, :use_k].copy_(V.t()[:hidden, :use_k])
            if proj.bias is not None:
                if mean_vec is not None:
                    proj.bias.data.copy_((-mean_vec @ V).to(proj.bias.dtype))
                else:
                    proj.bias.zero_()

MyEmbeddings = SpectraEmbeddings

