
"""Minimal example to verify PCA warm-start plumbing.

- Synthesizes 1D spectra (Gaussian absorption lines on noise)
- Tokenizes by chunking (patch_size=50, normalize per token)
- Estimates PCA on a few batches
- Initializes Q/K/V (+out) of a tiny HF ViT (for shape & API sanity)

If you don't have transformers installed, replace the model with your own ViT.
"""
import torch
from torch.utils.data import TensorDataset, DataLoader

try:
    from transformers import ViTConfig, ViTForImageClassification
    _HF = True
except Exception:
    _HF = False

from tokenizers_1d import Tokenizer1DConfig
from pca_warm import WarmStartConfig, pca_warm_start_model

def make_dummy_spectra(n=512, length=2000, seed=0):
    g = torch.Generator().manual_seed(seed)
    base = torch.randn(n, length, generator=g) * 0.05
    centers = [300, 600, 1200, 1600]
    for c in centers:
        x = torch.arange(length)
        line = torch.exp(-0.5*((x-c)/6.0)**2)[None, :]
        base -= 0.8*line
    return base

def main():
    spectra = make_dummy_spectra(512, 2000)
    dl = DataLoader(TensorDataset(spectra), batch_size=32, shuffle=True)

    if not _HF:
        print("Install 'transformers' to run the HF ViT demo, or swap in your ViT model.")
        return

    cfg = ViTConfig(image_size=32, patch_size=16, num_channels=3,
                    hidden_size=192, num_hidden_layers=4, num_attention_heads=3,
                    intermediate_size=768)
    model = ViTForImageClassification(cfg)

    tok_cfg  = Tokenizer1DConfig(mode='chunk', patch_size=50, normalize_chunk=True)
    warm_cfg = WarmStartConfig(n_components=192, share_QK=True, init_V_with_PCA=True,
                               per_head_rotate=True, zero_bias=True)

    sub = pca_warm_start_model(model, dl, tok_cfg, warm_cfg, max_batches=10, center=True, whiten=False)
    print("PCA components:", tuple(sub.components.shape), " first 5 EV:", 
          sub.explained_variance[:5].tolist() if sub.explained_variance is not None else None)
    # Show a Q weight shape for sanity
    for name, m in model.named_modules():
        if hasattr(m, 'attention') and hasattr(m.attention, 'attention'):
            print("[Sanity] Attention:", name, "Q.weight:", tuple(m.attention.attention.query.weight.shape))
            break

if __name__ == '__main__':
    main()
