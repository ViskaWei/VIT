
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
import numpy as np
import torch
try:
    import h5py
except Exception:
    h5py = None

from .core import fit_kpca_nystrom

def load_flux(path: str, key: str = "flux") -> torch.Tensor:
    p = Path(path)
    if p.suffix in [".npy", ".npz"]:
        arr = np.load(p, allow_pickle=False)
        if isinstance(arr, np.lib.npyio.NpzFile):
            X = arr[key] if key in arr else arr[list(arr.keys())[0]]
        else:
            X = arr
        return torch.from_numpy(np.asarray(X, dtype=np.float32))
    elif p.suffix in [".pt", ".pth"]:
        X = torch.load(p)
        return X.float()
    else:
        if h5py is None:
            raise RuntimeError("h5py not available to load HDF5")
        with h5py.File(p, "r") as f:
            if key not in f:
                raise KeyError(f"HDF5 missing dataset '{key}'. Available: {list(f.keys())}")
            X = f[key][:]
        return torch.from_numpy(np.asarray(X, dtype=np.float32))

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

    X = load_flux(args.data, key=args.dataset_key)
    if X.ndim != 2:
        raise ValueError(f"Expect [N, L], got {tuple(X.shape)}")
    print(f"[KPCA] Loaded {X.shape[0]} spectra, L={X.shape[1]}")

    gamma = None if args.gamma == "auto" else float(args.gamma)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        state = fit_kpca_nystrom(X.to(device), r=args.r, m_landmarks=args.landmarks,
                                  kernel_name=args.kernel, gamma=gamma,
                                  degree=args.degree, coef0=args.coef0,
                                  seed=args.seed, device=device)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        state.cpu().save(args.out)
        print(f"[KPCA] Saved state to {args.out}  (r={state.r}, landmarks={state.landmarks.shape[0]})")

if __name__ == "__main__":
    main()
