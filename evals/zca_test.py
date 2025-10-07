# zca_check.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Literal, Tuple
import torch

# -----------------------------
# 核心：协方差、收缩、ZCA、验证
# -----------------------------

def center(X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """按列中心化；返回 (Xc, mean)."""
    mu = X.mean(dim=0, keepdim=True)
    return X - mu, mu.squeeze(0)

def cov_N(Xc: torch.Tensor) -> torch.Tensor:
    """C = Xc^T Xc / N（与我们理论推导一致）"""
    N = Xc.shape[0]
    return (Xc.T @ Xc) / float(N)

def shrink_cov(C: torch.Tensor, gamma: float) -> torch.Tensor:
    """C_hat = (1-gamma) C + gamma * tr(C)/D * I"""
    if gamma <= 0.0:
        return C
    D = C.shape[0]
    tr_over_D = (torch.trace(C) / float(D)).to(C.dtype)
    return (1.0 - gamma) * C + gamma * tr_over_D * torch.eye(D, dtype=C.dtype, device=C.device)

def eigh_sorted_sym(C: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """对称化后做特征分解，并按降序排序。"""
    C = 0.5 * (C + C.T)  # 防数值不对称
    lam, V = torch.linalg.eigh(C)
    idx = torch.argsort(lam, descending=True)
    return lam[idx], V[:, idx]

def zca_from_cov(
    C_hat: torch.Tensor,
    eps: float = 1e-6,
    r: Optional[int] = None,
    perp_mode: Literal["zero", "identity", "avg"] = "identity",
    perp_scale: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    从 C_hat 构造 ZCA 矩阵 P，并返回 (P, projector, eigvals, eigvecs).
    P = V_r diag((lam_r+eps)^-1/2) V_r^T + s * (I - V_r V_r^T)
    注意：这里的实现严格等价于 V @ diag((lam+eps)^-1/2) @ V^T（全秩时）。
    """
    lam, V = eigh_sorted_sym(C_hat)
    D = lam.numel()

    if r is None or r <= 0 or r >= D:
        inv_sqrt = torch.rsqrt(lam + eps)
        P = (V * inv_sqrt) @ V.T  # 等价于 V @ diag(inv_sqrt) @ V^T（单边缩放）
        proj = V @ V.T            # = I
        return P, proj, lam, V

    # 低秩
    r = int(r)
    Vr = V[:, :r]
    lamr = lam[:r]
    inv_sqrt_r = torch.rsqrt(lamr + eps)
    P = (Vr * inv_sqrt_r) @ Vr.T
    proj = Vr @ Vr.T

    # 余空间尺度
    if perp_scale is None:
        if perp_mode == "identity":
            s = 1.0
        elif perp_mode == "avg" and r < D - 1:
            resid = lam[r:]
            s = float(torch.rsqrt(resid.mean() + eps))
        else:
            s = 0.0
    else:
        s = float(perp_scale)

    if s != 0.0:
        I = torch.eye(D, dtype=C_hat.dtype, device=C_hat.device)
        P = P + s * (I - proj)
    return P, proj, lam, V

# -----------------------------
# 验证指标
# -----------------------------

@torch.no_grad()
def rel_fro(A: torch.Tensor, B: torch.Tensor) -> float:
    return float(torch.linalg.norm(A - B, "fro") / torch.linalg.norm(B, "fro"))

@torch.no_grad()
def off_ratio(P: torch.Tensor) -> float:
    D = P.shape[0]
    return float(torch.linalg.norm(P - torch.diag(torch.diag(P)), "fro") / torch.linalg.norm(P, "fro"))

@dataclass
class ZCAReport:
    mu: torch.Tensor
    C: torch.Tensor
    C_hat: torch.Tensor
    P: torch.Tensor
    projector: torch.Tensor
    eigvals: torch.Tensor
    eigvecs: torch.Tensor
    sym_err: float
    white_err_self: float
    white_err_onX: float
    cond_C_hat: float
    cond_whitened: float
    off_ratio_P: float

def zca_pipeline(
    X: torch.Tensor,
    gamma: float = 0.1,
    eps: float = 1e-6,
    r: Optional[int] = None,
    perp_mode: Literal["zero", "identity", "avg"] = "identity",
    perp_scale: Optional[float] = None,
    device: Optional[torch.device] = None,
) -> ZCAReport:
    """
    主流程：从 X 得到 C、C_hat、P，并做三类验证。
    返回 ZCAReport（里含所有中间量，便于进一步分析/画图）。
    """
    X = X.detach()
    if device is None:
        device = X.device
    X = X.to(device=device, dtype=torch.float64)

    # 0) 中心化
    Xc, mu = center(X)

    # 1) 协方差（按 N）
    C = cov_N(Xc)

    # 2) 收缩（可选）
    C_hat = shrink_cov(C, gamma)

    # 3) ZCA（全秩或低秩）
    P, proj, lam, V = zca_from_cov(C_hat, eps=eps, r=r, perp_mode=perp_mode, perp_scale=perp_scale)

    # 4) 验证
    # 4.1 对称误差（ZCA 应近似对称）
    sym_err = rel_fro(P, P.T)

    # 4.2 自洽白化：P^T C_hat P 应≈ I
    I = torch.eye(C.shape[0], dtype=C.dtype, device=C.device)
    white_err_self = rel_fro(P.T @ C_hat @ P, I)

    # 4.3 用样本重算（同样的收缩）再验证
    #    注意：如果你校验时忘了应用同样的收缩，这里误差会放大
    Cx = cov_N(Xc)
    Cx_hat = shrink_cov(Cx, gamma)
    white_err_onX = rel_fro(P.T @ Cx_hat @ P, I)

    # 4.4 条件数：预条件化前后
    #    clip 防止低秩数值问题（仅用于 cond 估计，不影响上面验证）
    lam_clip = torch.clamp(lam, min=eps)
    cond_C_hat = float(lam_clip.max() / lam_clip.min())

    Ihat = P.T @ C_hat @ P
    lam2, _ = torch.linalg.eigh(0.5 * (Ihat + Ihat.T))
    lam2 = torch.clamp(lam2, min=eps)
    cond_whitened = float(lam2.max() / lam2.min())

    # 4.5 非对角能量占比（仅作风格参考）
    offp = off_ratio(P)

    return ZCAReport(
        mu=mu, C=C, C_hat=C_hat, P=P, projector=proj,
        eigvals=lam, eigvecs=V,
        sym_err=sym_err, white_err_self=white_err_self, white_err_onX=white_err_onX,
        cond_C_hat=cond_C_hat, cond_whitened=cond_whitened, off_ratio_P=offp
    )

# -----------------------------
# 一个可运行的例子（无噪声合成谱）
# -----------------------------
if __name__ == "__main__":
    # torch.set_printoptions(precision=4, sci_mode=True)

    # # 合成一个“无噪声但有相关性”的数据：
    # # X = S @ B^T，其中 B 是几个“谱基”（D=4096 维），S 是系数（N, r_true）
    # N, D, r_true = 4000, 512, 8
    # torch.manual_seed(0)
    # # 构造平滑的基（模拟 LSF 导致的邻域相关）
    # t = torch.linspace(0, 1, D)
    # centers = torch.linspace(0.1, 0.9, r_true)
    # widths = torch.linspace(0.02, 0.09, r_true)
    # B = torch.stack([torch.exp(-0.5*((t - c)/w)**2) for c, w in zip(centers, widths)], dim=1)  # [D, r_true]
    # # 让基向量正交化
    # B, _ = torch.linalg.qr(B, mode="reduced")
    # # 系数
    # S = torch.randn(N, r_true) @ torch.diag(torch.linspace(3.0, 1.0, r_true))
    # X = S @ B.T  # 无噪声合成谱：低秩+平滑相关
    import h5py
    import os
    num_samples = 10000
    TRAIN_DIR = os.getenv("TRAIN_DIR")
    with h5py.File(f"{TRAIN_DIR}/dataset.h5", "r") as f:
            X = torch.Tensor(f['dataset/arrays/flux/value'][:num_samples])
    print(f"X shape: {X.shape} (no noise)")
    
    rep = zca_pipeline(X, gamma=0.1, eps=1e-6, r=None, perp_mode="identity")
    print("\n=== ZCA REPORT (full-rank) ===")
    print(f"sym_err(P vs P^T):           {rep.sym_err:.3e}  (应极小)")
    print(f"white_err_self (P^T Ĉ P):   {rep.white_err_self:.3e}  (越小越好, <1e-1 合理)")
    print(f"white_err_onX  (by data):    {rep.white_err_onX:.3e}  (应与上面相近)")
    print(f"cond(Ĉ):                    {rep.cond_C_hat:.3e}  (通常很大)")
    print(f"cond(P^T Ĉ P):              {rep.cond_whitened:.3e}  (应接近 1)")
    print(f"off_ratio(P):                {rep.off_ratio_P:.3f}    (风格指标, 无硬门限)")

    # 低秩+余空间 identity
    rep_r = zca_pipeline(X, gamma=0.1, eps=1e-6, r=32, perp_mode="identity")
    print("\n=== ZCA REPORT (rank-32 + identity on perp) ===")
    print(f"white_err_self: {rep_r.white_err_self:.3e}, cond(P^T Ĉ P): {rep_r.cond_whitened:.3e}")
