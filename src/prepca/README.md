
# KPCA‑Warm for ViT (1D spectra)

This package provides:
1) Offline precompute of Kernel PCA using a Nyström approximation (works well for ~100k spectra).
2) A drop‑in PyTorch attention module `KPCAWarmSelfAttention` whose Q/K are built from KPCA features.

## Recommended workflow
1. **Precompute KPCA** on the clean 100k spectra (or on random patches). Save to `.pt`.
2. Use `KPCAWarmSelfAttention` in the first few Transformer blocks.
3. **Freeze Q/K adapters** for `freeze_qk_epochs` (e.g., 10–50), then unfreeze to fine‑tune.

## Precompute
```bash
python -m kpca_warm.precompute_kpca \
  --data /path/to/dataset.h5 \  --dataset-key flux \  --r 64 --landmarks 2048 \  --kernel rbf --gamma auto \  --out /path/to/kpca_state.pt
```
Notes:
- `--landmarks` controls Nyström subset size (e.g., 1024–4096).
- Ensure the vectors you fit KPCA **match the feature dim** seen by the attention layer
  where you will insert KPCA‑Warm (same last‑dim size). If they differ, fit KPCA on the
  vectors at that point (e.g., patch embeddings) or insert KPCA‑Warm accordingly.

## Use in code
```python
import torch
from kpca_warm import KernelPCAState, KPCAWarmSelfAttention

state = KernelPCAState.load("/path/to/kpca_state.pt").to(device)
attn  = KPCAWarmSelfAttention(dim=d_model, num_heads=H, kpca_state=state)
attn.set_qk_requires_grad(False)  # warm stage
# ... after warm epochs:
attn.set_qk_requires_grad(True)

# inside a Transformer block:
x = attn(x)  # x: [B, L, d_model]
```

## Centering
We store per-column means of K(L,L) and the global mean. For inference, `k(x,L)` is
centered using these stats, and projected with `A = eigvecs / sqrt(eigvals)`.

## Online updates?
Prefer **offline** KPCA for stability. For new domains/noise levels, recompute KPCA offline
or keep the clean KPCA and only fine‑tune the small Q/K adapters.


python -m src.kpca_warm.precompute_kpca --data $TEST_DIR/dataset.h5   --dataset-key dataset/arrays/flux/value  --r 1000  --kernel rbf --gamma auto  --out $PCA_DIR/kpca_state.pt



# robust_pcp.py

Robust PCA / Principal Component Pursuit (PCP) via ADMM with Singular Value Thresholding.
- Supports `svd='randomized'` using scikit-learn (fast for large, low-rank matrices).
- Falls back to full SVD if scikit-learn is not available.

## Usage
```python
import numpy as np
from robust_pcp import pcp

# X: shape (100_000, 4096), ideally float32 to save memory
# X = np.memmap('spectra.dat', dtype='float32', mode='r', shape=(100_000, 4096))

L, S, hist = pcp(
    X,
    svd='randomized',
    rank_guess=100,
    oversampling=20,
    power_iter=2,
    tol=1e-5,
    max_iter=500,
    dtype='float32',
    verbose=True
)
```

## Notes
- Default lambda = 1/sqrt(max(n, m)).
- If memory is tight, use `np.memmap` for X and keep `dtype='float32'`.