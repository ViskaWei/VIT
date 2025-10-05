# zca_utils.py
from __future__ import annotations
from typing import Optional, Literal, Tuple
import torch

def center(X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """列中心化；返回 (Xc, mean)。"""
    mu = X.mean(dim=0, keepdim=True)
    return X - mu, mu.squeeze(0)

def cov_N(Xc: torch.Tensor) -> torch.Tensor:
    """样本协方差 C = Xc^T Xc / N（与我们推导一致）"""
    N = Xc.shape[0]
    return (Xc.T @ Xc) / float(N)

def shrink_cov(C: torch.Tensor, gamma: float) -> torch.Tensor:
    """Ledoit–Wolf 风格收缩：Ĉ = (1-γ)C + γ·tr(C)/D·I"""
    if gamma <= 0.0:
        return C
    D = C.shape[0]
    tr_over_D = (torch.trace(C) / float(D)).to(C.dtype)
    I = torch.eye(D, dtype=C.dtype, device=C.device)
    return (1.0 - gamma) * C + gamma * tr_over_D * I

def eigh_sorted_sym(C: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """对称化 + 特征分解（降序）。"""
    C = 0.5 * (C + C.T)
    lam, V = torch.linalg.eigh(C)
    idx = torch.argsort(lam, descending=True)
    return lam[idx], V[:, idx]

def zca_from_cov(
    C_hat: torch.Tensor,
    eps: float = 1e-6,
    r: Optional[int] = None,
    perp_mode: Literal["zero", "identity", "avg"] = "identity",
    perp_scale: Optional[float] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    从 Ĉ 构造 ZCA 矩阵 P。
    全秩：P = V diag((λ+eps)^-1/2) V^T  （对称）
    低秩：P = V_r diag((λ_r+eps)^-1/2) V_r^T + s·(I - V_r V_r^T)
          s=1(identity) / 1/sqrt(avg λ_perp)(avg) / 0(zero)
    返回 (P, projector)；projector=I（全秩）或 V_r V_r^T（低秩）。
    """
    lam, V = eigh_sorted_sym(C_hat)
    D = lam.numel()

    if r is None or r <= 0 or r >= D:
        inv_sqrt = torch.rsqrt(lam + eps)
        P = (V * inv_sqrt) @ V.T  # 单边缩放，等价 V diag(inv_sqrt) V^T
        proj = V @ V.T            # = I
        return P, proj

    r = int(r)
    Vr = V[:, :r]
    inv_sqrt_r = torch.rsqrt(lam[:r] + eps)
    P = (Vr * inv_sqrt_r) @ Vr.T
    proj = Vr @ Vr.T

    if perp_scale is None:
        if perp_mode == "identity":
            s = 1.0
        elif perp_mode == "avg" and r < D - 1:
            s = float(torch.rsqrt(lam[r:].mean() + eps))
        else:
            s = 0.0
    else:
        s = float(perp_scale)
    if s != 0.0:
        I = torch.eye(D, dtype=C_hat.dtype, device=C_hat.device)
        P = P + s * (I - proj)

    return P, proj

@torch.no_grad()
def compute_zca_from_data(
    X: torch.Tensor,
    gamma: float = 0.1,
    eps: float = 1e-6,
    r: Optional[int] = None,
    perp_mode: str = "identity",
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    从数据 X 计算 (mu, Ĉ, P, projector)。
    - 先中心化 X；
    - Ĉ 用 /N 定义并做收缩；
    - 返回 P（对称）、以及一些中间量以便保存/复用。
    """
    if device is None: device = X.device
    X = X.to(device=device, dtype=dtype)
    Xc, mu = center(X)
    C = cov_N(Xc)
    C_hat = shrink_cov(C, gamma)
    P, proj = zca_from_cov(C_hat, eps=eps, r=r, perp_mode=perp_mode)
    return mu, C_hat, P, proj

# ---------- 验证工具 ----------
@torch.no_grad()
def rel_fro(A: torch.Tensor, B: torch.Tensor) -> float:
    return float(torch.linalg.norm(A - B, "fro") / torch.linalg.norm(B, "fro"))

@torch.no_grad()
def zca_report(
    X: torch.Tensor, mu: torch.Tensor, C_hat: torch.Tensor, P: torch.Tensor, gamma: float
) -> dict:
    I = torch.eye(P.shape[0], dtype=P.dtype, device=P.device)
    # 自洽白化：P^T Ĉ P ≈ I
    white_err_self = rel_fro(P.T @ C_hat @ P, I)
    # 用 X 重算（同样的收缩）再验一次
    X = X.to(dtype=P.dtype, device=P.device)
    Xc = X - mu.unsqueeze(0)
    Cx = cov_N(Xc)
    Cx_hat = shrink_cov(Cx, gamma)
    white_err_onX = rel_fro(P.T @ Cx_hat @ P, I)
    # 条件数
    lam, _ = torch.linalg.eigh(0.5 * (C_hat + C_hat.T))
    lam = torch.clamp(lam, min=1e-12)
    cond_C = float(lam.max() / lam.min())
    Ih = 0.5 * (P.T @ C_hat @ P + (P.T @ C_hat @ P).T)
    lam2, _ = torch.linalg.eigh(Ih)
    lam2 = torch.clamp(lam2, min=1e-12)
    cond_Ih = float(lam2.max() / lam2.min())
    # 对称性
    sym_err = rel_fro(P, P.T)
    return dict(
        sym_err=sym_err, white_err_self=white_err_self, white_err_onX=white_err_onX,
        cond_C_hat=cond_C, cond_whitened=cond_Ih
    )
