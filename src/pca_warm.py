
import warnings
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple, List, Union

import torch
import torch.nn as nn

from tokenizers_1d import Tokenizer1DConfig, tokenize_spectra

# ============================
# Utilities
# ============================

def _iter_spectra_batches(dataset_or_loader: Union[Iterable, torch.utils.data.Dataset], 
                          max_items: Optional[int] = None) -> Iterable[torch.Tensor]:
    seen = 0
    for item in dataset_or_loader:
        if isinstance(item, dict):
            x = item.get('spectrum', None)
            if x is None:
                for k in ['x', 'input', 'inputs', 'flux']:
                    if k in item:
                        x = item[k]; break
        elif isinstance(item, (list, tuple)):
            x = item[0]
        else:
            x = item
        if x is None:
            continue
        x = torch.as_tensor(x)
        if x.ndim == 1:
            x = x[None, :]
        yield x
        seen += len(x)
        if max_items is not None and seen >= max_items:
            return

def _orthonormal_matrix(d_out: int, d_in: int, device) -> torch.Tensor:
    a = torch.randn(d_in, d_out, device=device)
    q, _ = torch.linalg.qr(a, mode='reduced')  # (d_in, d_out) orthonormal cols
    return q.T  # (d_out, d_in)

# ============================
# PCA (Torch SVD)
# ============================

@dataclass
class Subspace:
    components: torch.Tensor  # (r, D) basis in token space (row-major)
    explained_variance: Optional[torch.Tensor] = None  # (r,)

def _torch_pca(X: torch.Tensor, n_components: int, center: bool = True, whiten: bool = False) -> Subspace:
    X = X.to(torch.float32)
    if center:
        X = X - X.mean(dim=0, keepdim=True)
    U, S, Vh = torch.linalg.svd(X, full_matrices=False)  # Vh: (D,D)
    r = min(n_components, Vh.shape[0])
    comps = Vh[:r, :]  # (r, D)
    n = X.shape[0]
    ev = (S**2) / max(1, (n - 1))
    ev = ev[:r]
    if whiten:
        sigma = S[:r].clamp_min(1e-12)
        comps = comps / sigma.unsqueeze(1)
    return Subspace(components=comps, explained_variance=ev)

def estimate_pca_subspace_from_dataset(dataset_or_loader: Union[Iterable, torch.utils.data.Dataset],
                                       tokenizer: Tokenizer1DConfig,
                                       max_batches: int = 50,
                                       n_components: Optional[int] = None,
                                       center: bool = True,
                                       whiten: bool = False) -> Subspace:
    feats = []
    batches = 0
    for x in _iter_spectra_batches(dataset_or_loader):
        tokens = tokenize_spectra(x, tokenizer)  # (B,N,P)
        B, N, P = tokens.shape
        feats.append(tokens.reshape(B*N, P))
        batches += 1
        if max_batches is not None and batches >= max_batches:
            break
    if not feats:
        raise RuntimeError("No spectra found for PCA warm-start.")
    X = torch.cat(feats, dim=0)
    r = n_components or min(64, X.shape[1])
    return _torch_pca(X, n_components=r, center=center, whiten=whiten)

# ============================
# Map token PCA -> hidden space
# ============================

def _infer_patch_linear_weight(model: nn.Module) -> Optional[torch.Tensor]:
    # Try typical attributes
    for name in ["patch_embed", "proj", "projection", "patch_projection", "token_embed"]:
        mod = getattr(model, name, None)
        if isinstance(mod, nn.Linear):
            return mod.weight.detach()
        if hasattr(mod, "proj") and isinstance(mod.proj, nn.Linear):
            return mod.proj.weight.detach()
        if hasattr(mod, "projection") and isinstance(mod.projection, nn.Linear):
            return mod.projection.weight.detach()
    # Search by name
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear) and any(k in name.lower() for k in ["patch", "proj", "embed"]):
            return m.weight.detach()
    return None

def _token_basis_to_hidden(basis_token: torch.Tensor, model: nn.Module, d_model: int) -> torch.Tensor:
    device = next(model.parameters()).device
    r, P = basis_token.shape
    W = _infer_patch_linear_weight(model)  # (d_model, P) if found
    if W is not None and W.shape[0] == d_model and W.shape[1] == P:
        U = (W @ basis_token.T).to(device)  # (d_model, r)
        q, _ = torch.linalg.qr(U, mode='reduced')
        return q
    if P == d_model:
        return basis_token.T.to(device)  # (d_model, r)
    warnings.warn(f"PCA basis dim (P={P}) != hidden dim (d_model={d_model}); using random orthonormal basis.")
    return _orthonormal_matrix(r, d_model, device).T  # (d_model, r)

# ============================
# QKV warm-start
# ============================

@dataclass
class WarmStartConfig:
    n_components: int
    apply_to_layers: Optional[List[int]] = None
    share_QK: bool = True
    init_V_with_PCA: bool = True
    zero_bias: bool = True
    per_head_rotate: bool = True
    pad_random_ortho: bool = True

def _distribute_components_over_heads(U: torch.Tensor, num_heads: int, head_dim: int) -> torch.Tensor:
    d_model, r = U.shape
    all_head = num_heads * head_dim
    out = torch.zeros(all_head, d_model, device=U.device)
    comp_per_head = max(1, (r + num_heads - 1)//num_heads)
    idx = 0
    for h in range(num_heads):
        start, end = h*head_dim, (h+1)*head_dim
        k = min(head_dim, comp_per_head, r - idx)
        if k > 0:
            out[start:start+k, :] = U[:, idx:idx+k].T
            idx += k
        if k < head_dim:
            out[start+k:end, :] = _orthonormal_matrix(head_dim-k, d_model, U.device)
    return out  # (all_head, d_model)

def _infer_heads(attn_module, q_lin: nn.Linear) -> Tuple[int, int, int]:
    all_head = q_lin.out_features
    d_model = q_lin.in_features
    if hasattr(attn_module, "num_attention_heads") and hasattr(attn_module, "attention_head_size"):
        nh = int(attn_module.num_attention_heads)
        hd = int(attn_module.attention_head_size)
        if nh * hd == all_head:
            return nh, hd, d_model
    # fallback guesses favoring divisibility
    for nh in [8, 6, 4, 3, 2]:
        if d_model % nh == 0 and all_head % nh == 0:
            hd = all_head // nh
            return nh, hd, d_model
    return 1, all_head, d_model

def build_qkv_from_pca(model: nn.Module,
                       pca_subspace: Subspace,
                       layer_indices: Optional[List[int]] = None,
                       share_QK: bool = True,
                       init_V_with_PCA: bool = True,
                       zero_bias: bool = True,
                       per_head_rotate: bool = True,
                       pad_random_ortho: bool = True) -> None:
    # Collect HF-style attention modules
    attn_sets = []  # (attn_mod, q_lin, k_lin, v_lin, out_lin)
    for name, module in model.named_modules():
        if hasattr(module, "attention") and hasattr(module.attention, "attention"):
            attn = module.attention.attention
            if all(hasattr(attn, x) for x in ["query", "key", "value"]):
                q_lin = attn.query; k_lin = attn.key; v_lin = attn.value
                out_lin = None
                if hasattr(module.attention, "output") and hasattr(module.attention.output, "dense"):
                    out_lin = module.attention.output.dense
                elif hasattr(attn, "out_proj"):
                    out_lin = attn.out_proj
                attn_sets.append((attn, q_lin, k_lin, v_lin, out_lin))
    if not attn_sets:
        warnings.warn("No HF-style attention modules found; adapt build_qkv_from_pca for your model.")
        return

    sample_q = attn_sets[0][1]
    d_model = sample_q.in_features
    comps_token = pca_subspace.components  # (r, P)
    U_hidden = _token_basis_to_hidden(comps_token, model, d_model)  # (d_model, r)
    r = U_hidden.shape[1]

    for li, (attn, q_lin, k_lin, v_lin, out_lin) in enumerate(attn_sets):
        if layer_indices is not None and li not in layer_indices:
            continue
        nh, hd, d_model = _infer_heads(attn, q_lin)
        all_head = nh * hd
        # per-head packing
        if per_head_rotate and nh > 1:
            W_base = _distribute_components_over_heads(U_hidden, nh, hd)
        else:
            if r < all_head:
                if pad_random_ortho:
                    pad = _orthonormal_matrix(all_head - r, d_model, U_hidden.device)
                    W_base = torch.cat([U_hidden.T, pad], dim=0)
                else:
                    reps = (all_head + r - 1)//r
                    W_base = torch.cat([U_hidden.T]*reps, dim=0)[:all_head, :]
            else:
                W_base = U_hidden.T[:all_head, :]

        with torch.no_grad():
            q_lin.weight.copy_(W_base)
            if share_QK:
                k_lin.weight.copy_(W_base)
            else:
                k_lin.weight.copy_(_orthonormal_matrix(all_head, d_model, W_base.device))
            if zero_bias:
                if q_lin.bias is not None: q_lin.bias.zero_()
                if k_lin.bias is not None: k_lin.bias.zero_()
            if init_V_with_PCA:
                v_lin.weight.copy_(W_base)
                if zero_bias and v_lin.bias is not None: v_lin.bias.zero_()
            if isinstance(out_lin, nn.Linear) and out_lin.in_features == all_head:
                out_lin.weight.copy_(W_base.T)  # (d_model, all_head)
                if zero_bias and out_lin.bias is not None: out_lin.bias.zero_()

def pca_warm_start_model(model: nn.Module,
                         dataset_or_loader: Union[Iterable, torch.utils.data.Dataset],
                         tokenizer_cfg: Tokenizer1DConfig,
                         warm_cfg: 'WarmStartConfig',
                         max_batches: int = 50,
                         center: bool = True,
                         whiten: bool = False) -> Subspace:
    """
    One-call warm start: estimate PCA -> init Q/K/V (+out) across layers.\n
    Returns Subspace (includes explained_variance).\n
    """
    sub = estimate_pca_subspace_from_dataset(dataset_or_loader, tokenizer=tokenizer_cfg,
                                             max_batches=max_batches, n_components=warm_cfg.n_components,
                                             center=center, whiten=whiten)
    build_qkv_from_pca(model, sub,
                       layer_indices=warm_cfg.apply_to_layers,
                       share_QK=warm_cfg.share_QK,
                       init_V_with_PCA=warm_cfg.init_V_with_PCA,
                       zero_bias=warm_cfg.zero_bias,
                       per_head_rotate=warm_cfg.per_head_rotate,
                       pad_random_ortho=warm_cfg.pad_random_ortho)
    return sub
