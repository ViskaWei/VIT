
"""CLI to precompute KPCA (Nyström) on 1D spectra and save to .pt

Example:
  python -m kpca_warm.precompute_kpca \
      --data /path/to/dataset.h5 \
      --dataset-key flux \
      --r 64 --landmarks 2048 \
      --kernel rbf --gamma auto \
      --out /path/to/kpca_state.pt
"""
import argparse
from pathlib import Path

import torch

from src.prepca.pipeline import compute_kernel_pca, load_spectra

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True, help="Path to HDF5/.npy/.pt with [N, L] spectra")
    ap.add_argument("--dataset-key", type=str, default="flux", help="HDF5 dataset key")
    ap.add_argument("--r", type=int, default=64, help="#KPCA components")
    ap.add_argument("--landmarks", type=int, default=2048, help="#landmarks for Nyström")
    ap.add_argument("--kernel", type=str, default="rbf", choices=["rbf", "poly", "linear"])
    ap.add_argument("--gamma", type=str, default="auto", help="'auto' or float")
    ap.add_argument("--degree", type=int, default=3, help="poly degree")
    ap.add_argument("--coef0", type=float, default=1.0, help="poly coef0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, required=True, help="Output .pt path")
    args = ap.parse_args()

    flux = load_spectra(args.data, dataset_key=args.dataset_key)["flux"]
    if flux.ndim != 2:
        raise ValueError(f"Expect [N, L], got {tuple(flux.shape)}")
    print(f"[KPCA] Loaded {flux.shape[0]} spectra, L={flux.shape[1]}")

    gamma = None if args.gamma == "auto" else float(args.gamma)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    state = compute_kernel_pca(
        flux.to(device),
        r=args.r,
        landmarks=args.landmarks,
        kernel_name=args.kernel,
        gamma=gamma,
        degree=args.degree,
        coef0=args.coef0,
        seed=args.seed,
        device=device,
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    state.cpu().save(args.out)
    print(f"[KPCA] Saved state to {args.out}  (r={state.r}, landmarks={state.landmarks.shape[0]})")

if __name__ == "__main__":
    main()
