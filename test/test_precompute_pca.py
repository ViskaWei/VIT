import os
import sys
import math
import torch

import types

from src.prepca.precompute_pca import build_patch_matrix, main as precompute_main


def test_pca_shapes_direct():
    """Compute PCA on synthetic patches and validate U, S, V shapes and orthonormality.

    Uses P.T as in the script: A = P.T with shape (D, M).
    """
    torch.manual_seed(0)
    # N, L, D = 1000, 40, 16  # ensure enough patches so M >= D # N = 100k, L = 4096, D = 32
    N, L, D = 1000, 32, 32  # ensure enough patches so M >= D # N = 100k, L = 4096, D = 32

    X = torch.randn(N, L)
    P, step = build_patch_matrix(X, patch=D)  # default non-overlap #()
    if L==D: assert (X==P).all(), "If L==D, P should equal X"

    M = P.shape[0]  #(N*(L-D)//D+1 if step==1 else N*(L//D))
    assert P.shape==(N*(L//D),D), f"Unexpected number of patches M: {M}"

    # PCA on transposed matrix to get component vectors of length D
    U, S, V = torch.pca_lowrank(P.T, q=D, center=True) #P.T = (D, M)
    print(f"[test] PCA P: {tuple(P.shape)} U: {tuple(U.shape)}, S: {tuple(S.shape)}, V: {tuple(V.shape)}")
    # Shape checks
    assert U.shape == (D, D), f"U shape {tuple(U.shape)} != (D, D)"
    assert S.shape == (D,), f"S shape {tuple(S.shape)} != (D,)"
    assert V.shape == (M, D), f"V shape {tuple(V.shape)} != (M, D)"

    # Orthonormality checks (columns are orthonormal)
    I = torch.eye(D, dtype=U.dtype)
    assert torch.allclose(U.T @ U, I, atol=1e-5), "U columns not orthonormal"
    assert torch.allclose(V.T @ V, I, atol=1e-5), "V columns not orthonormal"

    # Reconstruction of centered data
    A = P.T
    A_centered = A - A.mean(dim=0, keepdim=True)
    A_hat = U @ torch.diag(S) @ V.T
    rel_err = (A_centered - A_hat).norm() / (A_centered.norm() + 1e-12)
    assert rel_err < 1e-5, f"Reconstruction error too high: {rel_err}"


def test_precompute_cli_saves_expected_shapes(tmp_path, monkeypatch):
    """Run the CLI main() with a temporary dataset and verify saved tensor sizes.

    Ensures: U: (D, D), S: (D,), V: (M, D), and metadata is consistent.
    """
    torch.manual_seed(0)
    N, L, D = 100, 64, 16
    X = torch.randn(N, L)

    # Save synthetic data to a temporary .pt file
    data_path = os.path.join(tmp_path, "synthetic.pt")
    torch.save(X, data_path)

    # Route output into tmp dir and run main()
    monkeypatch.setenv("PCA_DIR", str(tmp_path))
    argv = [
        "precompute_pca",
        "--data", data_path,
        "--dataset-key", "ignored",
        "--patch-size", str(D),
        "--step", str(D),  # non-overlapping
        "--out", "pca_test.pt",
        "--device", "cpu",
    ]

    old_argv = sys.argv
    try:
        sys.argv = argv
        precompute_main()
    finally:
        sys.argv = old_argv

    out_path = os.path.join(tmp_path, "pca_test.pt")
    assert os.path.exists(out_path), "PCA output file not created"
    obj = torch.load(out_path)

    U = obj["U"]
    V = obj["V"]
    S = obj["S"]
    patch_size = int(obj["patch_size"])  # == D
    num_patches = int(obj["num_patches"])  # == M

    assert patch_size == D
    assert U.shape == (D, D), f"Saved U shape {tuple(U.shape)} != (D, D)"
    assert S.shape == (D,), f"Saved S shape {tuple(S.shape)} != (D,)"
    assert V.shape == (num_patches, D), f"Saved V shape {tuple(V.shape)} != (M, D)"

