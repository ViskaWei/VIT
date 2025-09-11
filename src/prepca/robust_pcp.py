
import numpy as np
from typing import Tuple, Dict, Any, Optional

try:
    from sklearn.utils.extmath import randomized_svd
    _HAS_SK = True
except Exception:
    _HAS_SK = False


def soft_threshold(X: np.ndarray, tau: float) -> np.ndarray:
    return np.sign(X) * np.maximum(np.abs(X) - tau, 0.0)


def svt_full(M: np.ndarray, tau: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    U, s, Vt = np.linalg.svd(M, full_matrices=False)
    s_thresh = np.maximum(s - tau, 0.0)
    return U, s_thresh, Vt


def svt_randomized(
    M: np.ndarray,
    tau: float,
    n_components: int,
    oversampling: int = 10,
    power_iter: int = 2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not _HAS_SK:
        return svt_full(M, tau)
    n_components = max(1, int(n_components))
    U, s, Vt = randomized_svd(
        M,
        n_components=n_components,
        n_oversamples=oversampling,
        n_iter=power_iter,
        transpose='auto',
        random_state=None,
    )
    s_thresh = np.maximum(s - tau, 0.0)
    nz = s_thresh > 0
    return U[:, nz], s_thresh[nz], Vt[nz, :]


def reconstruct_from_svd(U: np.ndarray, s: np.ndarray, Vt: np.ndarray) -> np.ndarray:
    if s.size == 0:
        return np.zeros((U.shape[0], Vt.shape[1]), dtype=U.dtype)
    return (U * s) @ Vt


def pcp(
    D: np.ndarray,
    lambda_: Optional[float] = None,
    mu: Optional[float] = None,
    tol: float = 1e-7,
    max_iter: int = 1000,
    svd: str = "randomized",
    rank_guess: int = 50,
    oversampling: int = 10,
    power_iter: int = 2,
    dtype: str = "float32",
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    D = np.asarray(D, dtype=dtype, order='C')
    n, m = D.shape
    normD = np.linalg.norm(D, ord='fro')
    if lambda_ is None:
        lambda_ = 1.0 / np.sqrt(max(n, m))
    L = np.zeros_like(D)
    S = np.zeros_like(D)
    Y = np.zeros_like(D)
    if mu is None:
        x = np.random.randn(m).astype(D.dtype)
        for _ in range(5):
            x = D.T @ (D @ x)
            x_norm = np.linalg.norm(x) + 1e-12
            x /= x_norm
        spectral = np.sqrt(np.linalg.norm(D @ x))
        mu = 1.25 / (spectral + 1e-12)
    mu_bar = mu * 1e7
    rho = 1.5
    history = {"primal_resid": [], "dual_resid": [], "rank": [], "nnz": [], "iters": 0}
    for k in range(1, max_iter + 1):
        M = D - S + (1.0 / mu) * Y
        if svd == "full":
            U, s, Vt = svt_full(M, 1.0 / mu)
        else:
            U, s, Vt = svt_randomized(M, 1.0 / mu, n_components=rank_guess, oversampling=oversampling, power_iter=power_iter)
        L = reconstruct_from_svd(U, s, Vt)
        r = D - L + (1.0 / mu) * Y
        S = soft_threshold(r, lambda_ / mu)
        R = D - L - S
        Y = Y + mu * R
        primal_resid = np.linalg.norm(R, ord='fro') / (normD + 1e-12)
        rank = int((s > 0).sum())
        nnz = int((np.abs(S) > 0).sum())
        history["primal_resid"].append(float(primal_resid))
        history["rank"].append(rank)
        history["nnz"].append(nnz)
        history["iters"] = k
        if verbose and (k % 10 == 0 or primal_resid < tol):
            print(f"[PCP] iter={k:4d}  resid={primal_resid:.3e}  rank(L)={rank}  nnz(S)={nnz}  mu={mu:.3e}")
        if primal_resid < tol:
            break
        mu = min(mu * rho, mu_bar)
    return L, S, history


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    n, m = 200, 80
    U_true = rng.standard_normal((n, 5))
    V_true = rng.standard_normal((5, m))
    L_true = U_true @ V_true
    S_true = np.zeros((n, m))
    mask = rng.random((n, m)) < 0.05
    S_true[mask] = rng.standard_normal(mask.sum()) * 5.0
    D = L_true + S_true
    L, S, hist = pcp(D, svd="randomized", rank_guess=10, oversampling=10, power_iter=1,
                     tol=1e-6, max_iter=500, dtype="float32", verbose=True)
    print("Final residual:", hist["primal_resid"][-1])
    print("Recovered rank(L):", hist["rank"][-1])
    print("Recovered nnz(S):", hist["nnz"][-1])
