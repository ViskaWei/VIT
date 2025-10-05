from __future__ import annotations

import torch
import torch.nn as nn

from .layers import ZCALinear


__all__ = ["GlobalAttentionLayer"]


class GlobalAttentionLayer(nn.Module):
    def __init__(
        self,
        input_dim,
        pca_stats,
        r: int | None = None,
        use_lora: bool = False,
        lora_min: int = 64,
        uv_key: str | None = None,
        use_pca_bias: bool = False,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.use_lora = bool(use_lora)
        self.r = int(r) if (r is not None) else None
        if self.use_lora:
            if self.r is None or self.r <= 0:
                self.r = min(lora_min, self.input_dim)
            rank = min(self.r, self.input_dim)
            self.rank = rank
            self.q_down = nn.Linear(input_dim, rank, bias=False)
            self.q_up = nn.Linear(rank, input_dim, bias=False)
            self.k_down = nn.Linear(input_dim, rank, bias=False)
            self.k_up = nn.Linear(rank, input_dim, bias=False)
            self.v_down = nn.Linear(input_dim, rank, bias=False)
            self.v_up = nn.Linear(rank, input_dim, bias=False)
        else:
            self.q_lin = nn.Linear(input_dim, input_dim, bias=False)
            self.k_lin = nn.Linear(input_dim, input_dim, bias=False)
            self.v_lin = nn.Linear(input_dim, input_dim, bias=False)

        self.softmax = nn.Softmax(dim=-1)
        self.preconditioner: ZCALinear | None = None
        self.input_bias = None

        if pca_stats is not None:
            self._init_from_pca(pca_stats, uv_key=uv_key, use_pca_bias=use_pca_bias)

    def _init_from_pca(self, pca_stats, uv_key: str | None, use_pca_bias: bool) -> None:
        if isinstance(pca_stats, dict):
            cand = None
            cand_key = None
            if isinstance(uv_key, str) and len(uv_key) > 0:
                key = uv_key.strip()
                if key.lower() in ("vt",):
                    pref = ["Vt", "Vh", "vh", "V", "components", "components_"]
                elif key.lower() in ("v",):
                    pref = ["V", "components", "components_", "Vt", "Vh", "vh"]
                elif key.lower() in ("ut",):
                    pref = ["Ut", "U"]
                elif key.lower() in ("u",):
                    pref = ["U", "Ut"]
                else:
                    pref = [key]
                for k in pref:
                    if k in pca_stats and isinstance(pca_stats[k], torch.Tensor):
                        cand = pca_stats[k]
                        cand_key = k
                        break
            if cand is None:
                for k in ("V", "components", "components_", "eigvecs", "eigvec", "basis", "Vh", "vh", "U"):
                    if k in pca_stats and isinstance(pca_stats[k], torch.Tensor):
                        cand = pca_stats[k]
                        cand_key = k
                        break
            if cand is not None:
                V = cand
                if V.dim() != 2:
                    raise ValueError("PCA basis must be 2D")
                if V.size(0) == self.input_dim and V.size(1) <= self.input_dim:
                    V_mat = V.contiguous()
                    inferred_r = V.size(1)
                elif V.size(1) == self.input_dim and V.size(0) <= self.input_dim:
                    V_mat = V.t().contiguous()
                    inferred_r = V.size(0)
                else:
                    raise ValueError(
                        f"Unexpected PCA basis shape {tuple(V.shape)} for input_dim={self.input_dim} (key={cand_key})"
                    )

                if self.use_lora:
                    use_r = min(self.rank, inferred_r)
                    V_mat = V_mat[:, :use_r]
                    self.q_down.weight.data.zero_()
                    self.q_up.weight.data.zero_()
                    self.k_down.weight.data.zero_()
                    self.k_up.weight.data.zero_()
                    self.q_down.weight.data[:use_r, :].copy_(V_mat.t())
                    self.q_up.weight.data[:, :use_r].copy_(V_mat)
                    self.k_down.weight.data[:use_r, :].copy_(V_mat.t())
                    self.k_up.weight.data[:, :use_r].copy_(V_mat)
                    nn.init.orthogonal_(self.v_down.weight)
                    nn.init.orthogonal_(self.v_up.weight)
                    try:
                        print(
                            f"[global-warmup] Basis key='{cand_key}', V_mat={tuple(V_mat.shape)}, applied_r={use_r}, mode='lora'"
                        )
                    except Exception:
                        pass
                else:
                    nn.init.orthogonal_(self.q_lin.weight)
                    nn.init.orthogonal_(self.k_lin.weight)
                    keep_r = min(inferred_r, self.r or inferred_r)
                    self.q_lin.weight.data[:keep_r, :].copy_(V_mat.t()[:keep_r, :])
                    self.k_lin.weight.data[:keep_r, :].copy_(V_mat.t()[:keep_r, :])
                    nn.init.orthogonal_(self.v_lin.weight)
                    try:
                        print(
                            f"[global-warmup] Basis key='{cand_key}', V_mat={tuple(V_mat.shape)}, applied_r={keep_r}, mode='full'"
                        )
                    except Exception:
                        pass

                self.explained_variance_at_r = None
                try:
                    if "explained_variance_ratio" in pca_stats:
                        evr = pca_stats["explained_variance_ratio"]
                        use_val = self.rank if self.use_lora else (self.r or evr.shape[0])
                        keep_r = min(int(use_val), int(evr.shape[0]))
                        self.explained_variance_at_r = float(evr[:keep_r].sum().item())
                        try:
                            print(f"[global-warmup] Explained variance@r ≈ {self.explained_variance_at_r:.4f}")
                        except Exception:
                            pass
                    elif "S" in pca_stats:
                        S = pca_stats["S"]
                        total_var = float((S ** 2).sum().item())
                        use_val = self.rank if self.use_lora else (self.r or S.shape[0])
                        keep_r = min(int(use_val), int(S.shape[0]))
                        self.explained_variance_at_r = float((S[:keep_r] ** 2).sum().item() / (total_var + 1e-12))
                        try:
                            print(f"[global-warmup] Explained variance@r ≈ {self.explained_variance_at_r:.4f}")
                        except Exception:
                            pass
                except Exception:
                    pass

            if bool(use_pca_bias):
                try:
                    m = pca_stats.get("mean", None)
                    if isinstance(m, torch.Tensor):
                        if m.dim() == 1 and int(m.numel()) == int(self.input_dim):
                            bias_vec = m.to(
                                device=(self.q_lin.weight.device if hasattr(self, "q_lin") else m.device),
                                dtype=(self.q_lin.weight.dtype if hasattr(self, "q_lin") else m.dtype),
                            )
                            self.register_buffer("input_bias", bias_vec, persistent=False)
                            print(f"[global-warmup] Loaded input bias from PCA mean: shape={tuple(bias_vec.shape)}")
                except Exception:
                    pass

            projector = None
            for key in ("projector", "projector_matrix", "zca", "preconditioner"):
                candidate = pca_stats.get(key) if isinstance(pca_stats, dict) else None
                if isinstance(candidate, torch.Tensor) and candidate.dim() == 2 and candidate.shape[1] == self.input_dim:
                    projector = candidate
                    break
            if projector is not None:
                try:
                    self.preconditioner = ZCALinear(projector, freeze=True)
                    print(
                        f"[global-warmup] Attached ZCALinear preconditioner with shape={tuple(projector.shape)}"
                    )
                except Exception as exc:
                    print(f"[global-warmup] Failed to attach ZCALinear preconditioner: {exc}")
        else:
            if isinstance(pca_stats, torch.Tensor) and pca_stats.dim() == 2 and pca_stats.shape[1] == self.input_dim:
                try:
                    self.preconditioner = ZCALinear(pca_stats, freeze=True)
                    print(
                        f"[global-warmup] Attached ZCALinear preconditioner (tensor) with shape={tuple(pca_stats.shape)}"
                    )
                except Exception as exc:
                    print(f"[global-warmup] Failed to attach ZCALinear preconditioner from tensor: {exc}")

    def q_proj(self, x):
        if self.use_lora:
            return self.q_up(self.q_down(x))
        return self.q_lin(x)

    def k_proj(self, x):
        if self.use_lora:
            return self.k_up(self.k_down(x))
        return self.k_lin(x)

    def v_proj(self, x):
        if self.use_lora:
            return self.v_up(self.v_down(x))
        return self.v_lin(x)

    def forward(self, x):
        if self.preconditioner is not None:
            x = self.preconditioner(x)
        if x.dim() == 2:
            ib = getattr(self, "input_bias", None)
            if isinstance(ib, torch.Tensor):
                x = x + ib
            return self.q_proj(x)

        ib = getattr(self, "input_bias", None)
        if isinstance(ib, torch.Tensor):
            x = x + ib
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.input_dim ** 0.5)
        attn_probs = self.softmax(attn_scores)
        attn_output = torch.matmul(attn_probs, v)
        return attn_output

    def set_qk_trainable(self, trainable: bool = True):
        if self.use_lora:
            modules = (self.q_down, self.q_up, self.k_down, self.k_up)
        else:
            modules = (self.q_lin, self.k_lin)
        for module in modules:
            for param in module.parameters():
                param.requires_grad = trainable
