from pathlib import Path

import h5py
import torch

from src.prepca.pipeline import (
    PreprocessingPipeline,
    compute_cka,
    compute_kernel_pca,
    compute_pca,
    compute_pcp,
    load_spectra,
    ensure_covariance,
)


def test_ensure_covariance_computes_and_persists(tmp_path):
    data = torch.arange(12, dtype=torch.float32).reshape(6, 2)
    cov_path = Path(tmp_path) / "cov.pt"

    stats_first = ensure_covariance(data, cov_path)
    assert cov_path.exists()

    perturbed = data + 100.0
    stats_second = ensure_covariance(perturbed, cov_path, allow_compute=False)

    assert torch.allclose(stats_first["cov"], stats_second["cov"])
    assert torch.allclose(stats_first["mean"], stats_second["mean"])


def test_compute_pca_shapes():
    spectra = torch.randn(16, 8)
    stats = compute_pca(spectra, patch_size=4, step=4, device=torch.device("cpu"))
    assert stats["components"].shape == (4, 4)
    assert int(stats["patch_size"].item()) == 4


def test_compute_kernel_pca_state_transform():
    spectra = torch.randn(10, 6)
    state = compute_kernel_pca(spectra, r=3, landmarks=5, device=torch.device("cpu"))
    transformed = state.transform(spectra)
    assert transformed.shape == (10, 3)


def test_compute_pcp_returns_arrays():
    spectra = torch.randn(6, 4)
    L, S, history = compute_pcp(spectra, max_iter=50, verbose=False)
    assert L.shape == (6, 4)
    assert S.shape == (6, 4)
    assert "iters" in history


def test_pipeline_run_cka(tmp_path):
    path = Path(tmp_path) / "cka.h5"
    flux = torch.randn(8, 5)
    with h5py.File(path, "w") as f:
        f.create_dataset("spectrumdataset/wave", data=torch.linspace(0, 1, 5).numpy())
        f.create_dataset("dataset/arrays/flux/value", data=flux.numpy())
    pipeline = PreprocessingPipeline(path)
    result = pipeline.run("cka", other=flux + 0.1)
    assert "cka" in result
    assert isinstance(result["cka"], torch.Tensor)


def test_load_spectra_reads_hdf5(tmp_path):
    path = Path(tmp_path) / "toy.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("spectrumdataset/wave", data=torch.arange(4).float().numpy())
        f.create_dataset("dataset/arrays/flux/value", data=torch.ones(3, 4).numpy())
        f.create_dataset("dataset/arrays/error/value", data=torch.zeros(3, 4).numpy())
    data = load_spectra(path)
    assert data["flux"].shape == (3, 4)
    assert data["wave"].shape == (4,)
    assert "error" in data
