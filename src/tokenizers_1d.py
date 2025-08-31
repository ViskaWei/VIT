
from dataclasses import dataclass
import torch

@dataclass
class Tokenizer1DConfig:
    mode: str = "chunk"  # "chunk" or "sliding"
    patch_size: int = 50
    overlap: int = 0
    normalize_chunk: bool = False

def chunkify_1d(x: torch.Tensor, patch_size: int) -> torch.Tensor:
    """x: (B, L) -> (B, N, patch_size), drop tail."""
    B, L = x.shape
    N = L // patch_size
    x = x[:, :N*patch_size]
    return x.reshape(B, N, patch_size)

def sliding_window_1d(x: torch.Tensor, window: int, overlap: int) -> torch.Tensor:
    """x: (B,L) -> (B, N, window) with step=window-overlap."""
    step = max(1, window - overlap)
    B, L = x.shape
    idx = torch.arange(0, max(L - window + 1, 1), step, device=x.device)
    patches = []
    for i in idx.tolist():
        patches.append(x[:, i:i+window].unsqueeze(1))
    if not patches:
        return x[:, :window].unsqueeze(1)
    return torch.cat(patches, dim=1)

def tokenize_spectra(x: torch.Tensor, cfg: Tokenizer1DConfig) -> torch.Tensor:
    if cfg.mode == "chunk":
        tokens = chunkify_1d(x, cfg.patch_size)
    elif cfg.mode == "sliding":
        tokens = sliding_window_1d(x, cfg.patch_size, cfg.overlap)
    else:
        raise ValueError(f"Unknown tokenizer mode: {cfg.mode}")
    if cfg.normalize_chunk:
        mu = tokens.mean(dim=-1, keepdim=True)
        std = tokens.std(dim=-1, keepdim=True).clamp_min(1e-6)
        tokens = (tokens - mu) / std
    return tokens
