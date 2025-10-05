import torch

from src.prepca.pipeline import ZCAWhitening


def _empirical_covariance(x: torch.Tensor) -> torch.Tensor:
    x = x - x.mean(dim=0)
    return x.t().matmul(x) / (x.shape[0] - 1)


def test_zca_whitening_produces_identity_covariance():
    torch.manual_seed(0)
    samples, features = 2048, 64
    scale = torch.linspace(1.0, 4.0, features, dtype=torch.float64)
    base = torch.randn(samples, features, dtype=torch.float64) * scale
    mix = torch.randn(features, features, dtype=torch.float64)
    cov_mix = mix @ mix.t()
    correlated = base.matmul(cov_mix)

    zca = ZCAWhitening(gamma=0.0, eps=1e-6, dtype=torch.float64)
    whitened = zca.fit_transform(correlated)
    cov = _empirical_covariance(whitened)
    identity = torch.eye(features, dtype=whitened.dtype)
    assert torch.allclose(cov, identity, atol=5e-2, rtol=5e-2)


def _generate_low_rank_regression(n_train=2048, n_test=512, features=64, rank=8):
    torch.manual_seed(1)
    dominant = torch.linspace(5.0, 3.0, rank, dtype=torch.float64)
    residual = torch.linspace(1.0, 0.2, features - rank, dtype=torch.float64)
    scales = torch.cat([dominant, residual])
    latent_train = torch.randn(n_train, features, dtype=torch.float64) * scales
    latent_test = torch.randn(n_test, features, dtype=torch.float64) * scales
    q, _ = torch.linalg.qr(torch.randn(features, features, dtype=torch.float64))
    observed_train = latent_train @ q
    observed_test = latent_test @ q
    weight = torch.randn(rank, 1, dtype=torch.float64)
    y_train = latent_train[:, :rank] @ weight
    y_test = latent_test[:, :rank] @ weight
    return observed_train, observed_test, y_train, y_test, rank, latent_train, latent_test, q


def _solve_linear(features: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    pseudo_inv = torch.linalg.pinv(features)
    return pseudo_inv @ targets


def _mae(pred: torch.Tensor, target: torch.Tensor) -> float:
    return float((pred - target).abs().mean().item())


def test_low_rank_projection_preserves_regression_quality():
    train_x, test_x, y_train, y_test, rank, latent_train, latent_test, rotation = _generate_low_rank_regression()

    full_zca = ZCAWhitening(gamma=0.0, eps=1e-6, dtype=torch.float64)
    train_full = full_zca.fit_transform(train_x)
    test_full = full_zca.transform(test_x)
    w_full = _solve_linear(train_full, y_train)
    pred_full = test_full @ w_full
    mae_full = _mae(pred_full, y_test)

    low_rank_zca = ZCAWhitening(gamma=0.0, eps=1e-6, rank=rank, alpha=0.0, dtype=torch.float64)
    low_rank_zca.fit(train_x)
    train_low = low_rank_zca.project(train_x)
    test_low = low_rank_zca.project(test_x)
    w_low = _solve_linear(train_low, y_train)
    pred_low = test_low @ w_low
    mae_low = _mae(pred_low, y_test)

    # Allow slight numerical slack (5%) to account for sampling noise
    assert mae_low <= mae_full * 1.05 + 1e-5
