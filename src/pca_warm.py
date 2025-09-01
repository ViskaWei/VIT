
# pca_warm.py
# =============================
from __future__ import annotations
import math
from typing import Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


def _center(X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Center features along batch dimension.
    Args:
        X: [n, d]
    Returns:
        Xc: centered X
        mu: mean vector [d]
    """
    mu = X.mean(dim=0, keepdim=True)
    return X - mu, mu.squeeze(0)


def fit_pca(
    X: torch.Tensor,
    r: int,
    robust: bool = False,
    kernel: Optional[str] = None,
    nystrom_m: Optional[int] = None,
    kernel_gamma: Optional[float] = None,
    ridge: float = 1e-6,
    device: Optional[torch.device] = None,
) -> Dict[str, torch.Tensor]:
    """
    Fit (kernel/robust) PCA on matrix X and return principal subspace.

    Args:
        X: [n, d] samples-as-rows (float32/float64). If robust=True, X is raw data.
        r: target rank (#components).
        robust: if True, do a simple PCP-like step (low-rank via truncated SVD) before PCA.
                (For production, plug an actual PCP solver; here we keep it light.)
        kernel: None | 'rbf' (Nyström KPCA if nystrom_m is not None, else full KPCA on minibatch).
        nystrom_m: # landmark samples for Nyström (<= n).
        kernel_gamma: RBF gamma (1/(2*sigma^2)). If None, use 1/dim heuristic.
        ridge: ridge added to Gram for numerical stability.
        device: optional device override.

    Returns dict with keys:
        'U': [d, r] principal directions in input space (for linear PCA),
        'eigvals': [r],
        'mu': [d], data mean,
        'proj': callable-like tuple (U, mu) for linear; for KPCA returns ('kmeans', anchors, alphas) style as tensors.
    Notes:
        - For RBF KPCA, we return an approximate linear projector using preimage-free feature maps:
          Phi(x) \approx k(x, C) K_CC^{-1/2}, then do PCA in that space and fold into an input-space linear map via least squares.
          This keeps the initialization simple for Q/K/V.
    """
    assert X.dim() == 2, "X should be [n, d]"
    n, d = X.shape
    dev = device or X.device

    X = X.to(dev)
    Xc, mu = _center(X)

    # Simple robust step: shrink outliers by Huber-like reweighting (placeholder for PCP)
    if robust:
        # scale by per-feature MAD to soften outliers; avoids full PCP dependency
        eps = 1e-6
        med = Xc.median(dim=0).values
        mad = (Xc - med).abs().median(dim=0).values + eps
        w = 1.0 / (1.0 + (Xc - med).abs() / (3.0 * mad))  # in (0,1]
        Xc = w * Xc

    if kernel is None:
        # Linear PCA via SVD on centered data
        # Xc = U S V^T, columns of V are principal axes in feature space
        # We want U_lin = V[:, :r]
        U, S, Vh = torch.linalg.svd(Xc, full_m=False)
        V = Vh.transpose(0, 1)
        U_lin = V[:, :r].contiguous()
        eigvals = (S[:r] ** 2) / max(n - 1, 1)
        return {"U": U_lin, "eigvals": eigvals, "mu": mu}

    # Kernel PCA (RBF) with Nyström approximation
    assert kernel == 'rbf', "Only 'rbf' is supported for kernel PCA in this skeleton"
    m = nystrom_m or min(n, 512)
    # Choose anchor subset
    perm = torch.randperm(n, device=dev)[:m]
    C = Xc[perm]  # [m, d]

    # gamma heuristic
    if kernel_gamma is None:
        # median heuristic on anchors (approx)
        with torch.no_grad():
            d2 = torch.cdist(C, C) ** 2
            med2 = d2.median()
        kernel_gamma = 0.5 / (med2 + 1e-8)

    def rbf(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.exp(-kernel_gamma * torch.cdist(a, b) ** 2)

    K_CC = rbf(C, C)
    # Regularize
    K_CC = K_CC + ridge * torch.eye(m, device=dev)
    # Eigendecompose
    Scc, Ucc = torch.linalg.eigh(K_CC)  # ascending
    idx = torch.argsort(Scc, descending=True)
    Scc = Scc[idx]
    Ucc = Ucc[:, idx]

    # Form feature map: Phi(x) ≈ k(x,C) * Ucc[:, :r] * Scc[:r]^{-1/2}
    Sr = torch.clamp(Scc[:r], min=ridge)
    B = Ucc[:, :r] / torch.sqrt(Sr)[None, :]

    # Build a linear projector W s.t. W^T x ≈ PCA in Phi(x). Fit W by LS: Xc W ≈ Z, with Z = Phi(Xc)
    K_XC = rbf(Xc, C)  # [n, m]
    Z = K_XC @ B  # [n, r]
    # Solve min_W ||Xc W - Z||^2 + ridge ||W||^2  -> W = (Xc^T Xc + λI)^{-1} Xc^T Z
    XtX = Xc.T @ Xc
    W = torch.linalg.solve(XtX + ridge * torch.eye(d, device=dev), Xc.T @ Z)  # [d, r]

    eigvals = (Z.var(dim=0, unbiased=True))  # proxy
    return {"U": W, "eigvals": eigvals, "mu": mu}


@torch.no_grad()
def init_qkv_from_pca(
    module: nn.Module,
    U: torch.Tensor,
    mu: Optional[torch.Tensor] = None,
    per_head: bool = True,
    whiten: bool = False,
    eigvals: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """
    Initialize Q/K/V (and optional output proj) from PCA/KPCA directions.

    Supports:
    - MultiheadAttention (PyTorch): q_proj_weight, k_proj_weight, v_proj_weight, in_proj_weight, out_proj
    - ViT-style fused qkv: a Linear named 'qkv' with out_features = 3 * d_model
    - Separate q_proj/k_proj/v_proj Linear layers commonly used in custom ViT blocks

    Args:
        module: an attention block or parent layer that *contains* q/k/v (or fused qkv) linears.
        U: [d_model, r] principal directions mapped to model input dim.
        mu: optional mean (unused here; kept for future centering layers).
        per_head: if True, split U columns evenly into heads; else tile/orth-complete per head.
        whiten: if True and eigvals provided, scale columns by 1/sqrt(eigval).
        eigvals: [r] used when whiten=True.

    Returns: dict of actually assigned weight tensors for logging.
    """
    assigned = {}

    # Try to discover qkv parameters
    def _find_linear(name_candidates):
        for name in name_candidates:
            mod = dict(module.named_modules()).get(name, None)
            if isinstance(mod, nn.Linear):
                return name, mod
        # also look in attributes directly
        for name in name_candidates:
            mod = getattr(module, name, None)
            if isinstance(mod, nn.Linear):
                return name, mod
        return None, None

    # Attempt to infer dims
    d_model = None
    num_heads = getattr(module, 'num_heads', None)
    head_dim = getattr(module, 'head_dim', None)

    # Common patterns
    name_qkv, lin_qkv = _find_linear(['qkv'])
    name_q, lin_q = _find_linear(['q', 'q_proj', 'q_linear'])
    name_k, lin_k = _find_linear(['k', 'k_proj', 'k_linear'])
    name_v, lin_v = _find_linear(['v', 'v_proj', 'v_linear'])
    name_o, lin_o = _find_linear(['proj', 'out_proj', 'o', 'o_proj'])

    if lin_qkv is not None:
        out_dim, in_dim = lin_qkv.weight.shape
        assert out_dim % 3 == 0, "qkv out dim should be 3*d_model"
        d_model = out_dim // 3
        if num_heads is None:
            # heuristic: try to infer from head_dim if present elsewhere
            num_heads = getattr(module, 'nhead', None) or getattr(module, 'num_heads', 1)
        if head_dim is None and num_heads:
            head_dim = d_model // num_heads

        U_use = U
        if whiten and eigvals is not None:
            scale = 1.0 / torch.sqrt(torch.clamp(eigvals, min=1e-8))
            U_use = U_use @ torch.diag(scale.to(U.device))

        if per_head and num_heads and head_dim:
            r = U_use.shape[1]
            # distribute columns across heads (truncate or tile as needed)
            cols_per_head = min(head_dim, max(1, r // num_heads))
            U_blocks = []
            for h in range(num_heads):
                start = h * cols_per_head
                end = min(start + cols_per_head, r)
                Uh = U_use[:, start:end]
                if Uh.shape[1] < head_dim:
                    # orth-complete within span (pad with zeros)
                    pad = torch.zeros(U_use.shape[0], head_dim - Uh.shape[1], device=U_use.device)
                    Uh = torch.cat([Uh, pad], dim=1)
                U_blocks.append(Uh)
            U_hcat = torch.cat(U_blocks, dim=1)  # [d_model, num_heads*head_dim]
            # Assign to q/k/v slices inside fused qkv
            with torch.no_grad():
                lin_qkv.weight[:d_model, :] = U_hcat.T
                lin_qkv.weight[d_model:2*d_model, :] = U_hcat.T
                lin_qkv.weight[2*d_model:, :] = U_hcat.T
            assigned['qkv.weight'] = lin_qkv.weight.detach().clone()
        else:
            with torch.no_grad():
                lin_qkv.weight[:d_model, :] = U_use.T
                lin_qkv.weight[d_model:2*d_model, :] = U_use.T
                lin_qkv.weight[2*d_model:, :] = U_use.T
            assigned['qkv.weight'] = lin_qkv.weight.detach().clone()

        if lin_o is not None:
            with torch.no_grad():
                lin_o.weight.copy_(U.T)
            assigned['out_proj.weight'] = lin_o.weight.detach().clone()
        return assigned

    # Separate q/k/v lines
    if lin_q is not None and lin_k is not None and lin_v is not None:
        _, in_dim = lin_q.weight.shape
        d_model = in_dim
        U_use = U
        if whiten and eigvals is not None:
            scale = 1.0 / torch.sqrt(torch.clamp(eigvals, min=1e-8))
            U_use = U_use @ torch.diag(scale.to(U.device))

        with torch.no_grad():
            lin_q.weight.copy_(U_use.T)
            lin_k.weight.copy_(U_use.T)
            lin_v.weight.copy_(U_use.T)
        assigned['q.weight'] = lin_q.weight.detach().clone()
        assigned['k.weight'] = lin_k.weight.detach().clone()
        assigned['v.weight'] = lin_v.weight.detach().clone()
        if lin_o is not None:
            with torch.no_grad():
                lin_o.weight.copy_(U.T)
            assigned['out_proj.weight'] = lin_o.weight.detach().clone()
        return assigned

    raise RuntimeError("Could not locate qkv parameters in the provided module.")


# import warnings
# from dataclasses import dataclass
# from typing import Iterable, Optional, Tuple, List, Union

# import torch
# import torch.nn as nn

# from src.tokenizers_1d import Tokenizer1DConfig, tokenize_spectra

# # ============================
# # Utilities
# # ============================

# def _iter_spectra_batches(dataset_or_loader: Union[Iterable, torch.utils.data.Dataset], 
#                           max_items: Optional[int] = None) -> Iterable[torch.Tensor]:
#     seen = 0
#     for item in dataset_or_loader:
#         if isinstance(item, dict):
#             x = item.get('spectrum', None)
#             if x is None:
#                 for k in ['x', 'input', 'inputs', 'flux']:
#                     if k in item:
#                         x = item[k]; break
#         elif isinstance(item, (list, tuple)):
#             x = item[0]
#         else:
#             x = item
#         if x is None:
#             continue
#         x = torch.as_tensor(x)
#         if x.ndim == 1:
#             x = x[None, :]
#         yield x
#         seen += len(x)
#         if max_items is not None and seen >= max_items:
#             return

# def _orthonormal_matrix(d_out: int, d_in: int, device) -> torch.Tensor:
#     a = torch.randn(d_in, d_out, device=device)
#     q, _ = torch.linalg.qr(a, mode='reduced')  # (d_in, d_out) orthonormal cols
#     return q.T  # (d_out, d_in)

# # ============================
# # PCA (Torch SVD)
# # ============================

# @dataclass
# class Subspace:
#     components: torch.Tensor  # (r, D) basis in token space (row-major)
#     explained_variance: Optional[torch.Tensor] = None  # (r,)

# def _torch_pca(X: torch.Tensor, n_components: int, center: bool = True, whiten: bool = False) -> Subspace:
#     X = X.to(torch.float32)
#     if center:
#         X = X - X.mean(dim=0, keepdim=True)
#     U, S, Vh = torch.linalg.svd(X, full_matrices=False)  # Vh: (D,D)
#     r = min(n_components, Vh.shape[0])
#     comps = Vh[:r, :]  # (r, D)
#     n = X.shape[0]
#     ev = (S**2) / max(1, (n - 1))
#     ev = ev[:r]
#     if whiten:
#         sigma = S[:r].clamp_min(1e-12)
#         comps = comps / sigma.unsqueeze(1)
#     return Subspace(components=comps, explained_variance=ev)

# def estimate_pca_subspace_from_dataset(dataset_or_loader: Union[Iterable, torch.utils.data.Dataset],
#                                        tokenizer: Tokenizer1DConfig,
#                                        max_batches: int = 50,
#                                        n_components: Optional[int] = None,
#                                        center: bool = True,
#                                        whiten: bool = False) -> Subspace:
#     feats = []
#     batches = 0
#     for x in _iter_spectra_batches(dataset_or_loader):
#         tokens = tokenize_spectra(x, tokenizer)  # (B,N,P)
#         B, N, P = tokens.shape
#         feats.append(tokens.reshape(B*N, P))
#         batches += 1
#         if max_batches is not None and batches >= max_batches:
#             break
#     if not feats:
#         raise RuntimeError("No spectra found for PCA warm-start.")
#     X = torch.cat(feats, dim=0)
#     r = n_components or min(64, X.shape[1])
#     return _torch_pca(X, n_components=r, center=center, whiten=whiten)

# # ============================
# # Map token PCA -> hidden space
# # ============================

# def _infer_patch_linear_weight(model: nn.Module) -> Optional[torch.Tensor]:
#     # Try typical attributes
#     for name in ["patch_embed", "proj", "projection", "patch_projection", "token_embed"]:
#         mod = getattr(model, name, None)
#         if isinstance(mod, nn.Linear):
#             return mod.weight.detach()
#         if hasattr(mod, "proj") and isinstance(mod.proj, nn.Linear):
#             return mod.proj.weight.detach()
#         if hasattr(mod, "projection") and isinstance(mod.projection, nn.Linear):
#             return mod.projection.weight.detach()
#     # Search by name
#     for name, m in model.named_modules():
#         if isinstance(m, nn.Linear) and any(k in name.lower() for k in ["patch", "proj", "embed"]):
#             return m.weight.detach()
#     return None

# def _token_basis_to_hidden(basis_token: torch.Tensor, model: nn.Module, d_model: int) -> torch.Tensor:
#     device = next(model.parameters()).device
#     r, P = basis_token.shape
#     W = _infer_patch_linear_weight(model)  # (d_model, P) if found
#     if W is not None and W.shape[0] == d_model and W.shape[1] == P:
#         U = (W @ basis_token.T).to(device)  # (d_model, r)
#         q, _ = torch.linalg.qr(U, mode='reduced')
#         return q
#     if P == d_model:
#         return basis_token.T.to(device)  # (d_model, r)
#     warnings.warn(f"PCA basis dim (P={P}) != hidden dim (d_model={d_model}); using random orthonormal basis.")
#     return _orthonormal_matrix(r, d_model, device).T  # (d_model, r)

# # ============================
# # QKV warm-start
# # ============================

# @dataclass
# class WarmStartConfig:
#     n_components: int
#     apply_to_layers: Optional[List[int]] = None
#     share_QK: bool = True
#     init_V_with_PCA: bool = True
#     zero_bias: bool = True
#     per_head_rotate: bool = True
#     pad_random_ortho: bool = True

# def _distribute_components_over_heads(U: torch.Tensor, num_heads: int, head_dim: int) -> torch.Tensor:
#     d_model, r = U.shape
#     all_head = num_heads * head_dim
#     out = torch.zeros(all_head, d_model, device=U.device)
#     comp_per_head = max(1, (r + num_heads - 1)//num_heads)
#     idx = 0
#     for h in range(num_heads):
#         start, end = h*head_dim, (h+1)*head_dim
#         k = min(head_dim, comp_per_head, r - idx)
#         if k > 0:
#             out[start:start+k, :] = U[:, idx:idx+k].T
#             idx += k
#         if k < head_dim:
#             out[start+k:end, :] = _orthonormal_matrix(head_dim-k, d_model, U.device)
#     return out  # (all_head, d_model)

# def _infer_heads(attn_module, q_lin: nn.Linear) -> Tuple[int, int, int]:
#     all_head = q_lin.out_features
#     d_model = q_lin.in_features
#     if hasattr(attn_module, "num_attention_heads") and hasattr(attn_module, "attention_head_size"):
#         nh = int(attn_module.num_attention_heads)
#         hd = int(attn_module.attention_head_size)
#         if nh * hd == all_head:
#             return nh, hd, d_model
#     # fallback guesses favoring divisibility
#     for nh in [8, 6, 4, 3, 2]:
#         if d_model % nh == 0 and all_head % nh == 0:
#             hd = all_head // nh
#             return nh, hd, d_model
#     return 1, all_head, d_model

# def build_qkv_from_pca(model: nn.Module,
#                        pca_subspace: Subspace,
#                        layer_indices: Optional[List[int]] = None,
#                        share_QK: bool = True,
#                        init_V_with_PCA: bool = True,
#                        zero_bias: bool = True,
#                        per_head_rotate: bool = True,
#                        pad_random_ortho: bool = True) -> None:
#     # Collect HF-style attention modules
#     attn_sets = []  # (attn_mod, q_lin, k_lin, v_lin, out_lin)
#     for name, module in model.named_modules():
#         if hasattr(module, "attention") and hasattr(module.attention, "attention"):
#             attn = module.attention.attention
#             if all(hasattr(attn, x) for x in ["query", "key", "value"]):
#                 q_lin = attn.query; k_lin = attn.key; v_lin = attn.value
#                 out_lin = None
#                 if hasattr(module.attention, "output") and hasattr(module.attention.output, "dense"):
#                     out_lin = module.attention.output.dense
#                 elif hasattr(attn, "out_proj"):
#                     out_lin = attn.out_proj
#                 attn_sets.append((attn, q_lin, k_lin, v_lin, out_lin))
#     if not attn_sets:
#         warnings.warn("No HF-style attention modules found; adapt build_qkv_from_pca for your model.")
#         return

#     sample_q = attn_sets[0][1]
#     d_model = sample_q.in_features
#     comps_token = pca_subspace.components  # (r, P)
#     U_hidden = _token_basis_to_hidden(comps_token, model, d_model)  # (d_model, r)
#     r = U_hidden.shape[1]

#     for li, (attn, q_lin, k_lin, v_lin, out_lin) in enumerate(attn_sets):
#         if layer_indices is not None and li not in layer_indices:
#             continue
#         nh, hd, d_model = _infer_heads(attn, q_lin)
#         all_head = nh * hd
#         # per-head packing
#         if per_head_rotate and nh > 1:
#             W_base = _distribute_components_over_heads(U_hidden, nh, hd)
#         else:
#             if r < all_head:
#                 if pad_random_ortho:
#                     pad = _orthonormal_matrix(all_head - r, d_model, U_hidden.device)
#                     W_base = torch.cat([U_hidden.T, pad], dim=0)
#                 else:
#                     reps = (all_head + r - 1)//r
#                     W_base = torch.cat([U_hidden.T]*reps, dim=0)[:all_head, :]
#             else:
#                 W_base = U_hidden.T[:all_head, :]

#         with torch.no_grad():
#             q_lin.weight.copy_(W_base)
#             if share_QK:
#                 k_lin.weight.copy_(W_base)
#             else:
#                 k_lin.weight.copy_(_orthonormal_matrix(all_head, d_model, W_base.device))
#             if zero_bias:
#                 if q_lin.bias is not None: q_lin.bias.zero_()
#                 if k_lin.bias is not None: k_lin.bias.zero_()
#             if init_V_with_PCA:
#                 v_lin.weight.copy_(W_base)
#                 if zero_bias and v_lin.bias is not None: v_lin.bias.zero_()
#             if isinstance(out_lin, nn.Linear) and out_lin.in_features == all_head:
#                 out_lin.weight.copy_(W_base.T)  # (d_model, all_head)
#                 if zero_bias and out_lin.bias is not None: out_lin.bias.zero_()

# def pca_warm_start_model(model: nn.Module,
#                          dataset_or_loader: Union[Iterable, torch.utils.data.Dataset],
#                          tokenizer_cfg: Tokenizer1DConfig,
#                          warm_cfg: 'WarmStartConfig',
#                          max_batches: int = 50,
#                          center: bool = True,
#                          whiten: bool = False) -> Subspace:
#     """
#     One-call warm start: estimate PCA -> init Q/K/V (+out) across layers.\n
#     Returns Subspace (includes explained_variance).\n
#     """
#     sub = estimate_pca_subspace_from_dataset(dataset_or_loader, tokenizer=tokenizer_cfg,
#                                              max_batches=max_batches, n_components=warm_cfg.n_components,
#                                              center=center, whiten=whiten)
#     build_qkv_from_pca(model, sub,
#                        layer_indices=warm_cfg.apply_to_layers,
#                        share_QK=warm_cfg.share_QK,
#                        init_V_with_PCA=warm_cfg.init_V_with_PCA,
#                        zero_bias=warm_cfg.zero_bias,
#                        per_head_rotate=warm_cfg.per_head_rotate,
#                        pad_random_ortho=warm_cfg.pad_random_ortho)
#     return sub
