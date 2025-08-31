
import torch

def _center_gram(K: torch.Tensor) -> torch.Tensor:
    n = K.shape[0]
    one_n = torch.full((n, n), 1.0 / n, dtype=K.dtype, device=K.device)
    return K - one_n @ K - K @ one_n + one_n @ K @ one_n

def linear_CKA(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    \"\"\"Linear CKA between (n,d1) and (n,d2). Returns scalar tensor.\"\"\"
    if X.ndim > 2: X = X.flatten(1)
    if Y.ndim > 2: Y = Y.flatten(1)
    Kx = X @ X.T
    Ky = Y @ Y.T
    Kxc = _center_gram(Kx)
    Kyc = _center_gram(Ky)
    num = (Kxc * Kyc).sum()
    den = torch.linalg.norm(Kxc) * torch.linalg.norm(Kyc) + 1e-12
    return num / den
