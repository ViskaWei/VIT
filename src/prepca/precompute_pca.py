"""CLI to precompute PCA basis for 1D patch embeddings and save to .pt

This computes PCA over length-`patch_size` windows taken from spectra, then saves
principal components under key 'V' so they can initialize the linear patch
projection in the embedding layer.

Examples:
  python -m src.prepca.precompute_pca \
      --data ${TRAIN_DIR}/dataset.h5 \
      --dataset-key dataset/arrays/flux/value \
      --patch-size 32 \
      --out pca_patch_32.pt
"""

import argparse
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch

try:
    import h5py  # type: ignore
except Exception:
    h5py = None  # h5 not required if using npy/pt


def load_flux_any(path: str, key: str) -> torch.Tensor:
    """Load spectra matrix X with shape [N, L] from HDF5/.npy/.pt files.

    - For .h5/.hdf5: reads dataset under 'key'.
    - For .npy/.npz: loads array; for .npz, use 'key' if present else first.
    - For .pt/.pth: loads tensor and converts to float32.
    """
    p = Path(path)
    suf = p.suffix.lower()
    if suf in (".npy", ".npz"):
        arr = np.load(p, allow_pickle=False)
        if isinstance(arr, np.lib.npyio.NpzFile):
            X = arr[key] if key in arr else arr[list(arr.keys())[0]]
        else:
            X = arr
        X = np.asarray(X, dtype=np.float32)
        return torch.from_numpy(X)
    if suf in (".pt", ".pth"):
        X = torch.load(p)
        return X.float()
    # HDF5
    if h5py is None:
        raise RuntimeError("h5py is not available to load HDF5 files")
    with h5py.File(p, "r") as f:
        if key not in f:
            # try a few fallbacks commonly seen
            candidates = [
                key,
                "flux",
                "dataset/arrays/flux/value",
                "dataset/arrays/flux",
            ]
            found = None
            for k in candidates:
                if k in f:
                    found = k
                    break
            if found is None:
                raise KeyError(f"Dataset key '{key}' not found in HDF5. Available top-level: {list(f.keys())}")
            key = found
        X = f[key][:]
    X = np.asarray(X, dtype=np.float32)
    return torch.from_numpy(X)


def build_patch_matrix(X: torch.Tensor, patch: int, step: Optional[int] = None) -> torch.Tensor:
    """Extract length-`patch` windows from each row and stack to [M, patch].

    By default uses non-overlapping windows (step == patch). Use `step` to change
    the stride (e.g., 1 for fully overlapping).
    """
    if X.ndim != 2:
        raise ValueError(f"Expect [N, L], got {tuple(X.shape)}")
    step = int(step) if (step is not None and int(step) > 0) else patch
    # X: [N, L] -> [N, num_patches, patch]
    patches = X.unfold(1, patch, step)
    # Flatten to [M, patch]
    P = patches.contiguous().view(-1, patch)
    return P, step


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=False,
                    default=os.environ.get("TRAIN_DIR", "./data") + "/dataset.h5",
                    help="Path to HDF5/.npy/.pt with [N, L] spectra")
    ap.add_argument("--dataset-key", type=str, default="dataset/arrays/flux/value",
                    help="Dataset key for HDF5 or .npz")
    ap.add_argument("--patch-size", type=int, required=True, help="Patch length (D)")
    ap.add_argument("--step", type=int, default=None, help="Stride between windows (default=patch-size)")
    ap.add_argument("--limit", type=int, default=None, help="Limit number of spectra (rows)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--out", type=str, default=None, help="Output .pt path; default=pca_patch_{patch}.pt")
    ap.add_argument("--plot", action="store_true", help="Also save spectrum and top components plots next to out")
    args = ap.parse_args()

    torch.manual_seed(int(args.seed))

    X = load_flux_any(args.data, key=args.dataset_key)
    if args.limit is not None and int(args.limit) > 0 and X.shape[0] > int(args.limit):
        X = X[: int(args.limit)]
    print(f"[PCA] Loaded X: {tuple(X.shape)}")

    P, step = build_patch_matrix(X, patch=args.patch_size, step=args.step)
    print(f"[PCA] Patch matrix: {tuple(P.shape)} (M x D)")

    dev = (torch.device("cuda") if (args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()))
           else torch.device("cpu"))
    P = P.to(dev)

    D = int(args.patch_size)
    with torch.no_grad():
        # Compute up to D components; center=True gives PCA basis in V
        U, S, V = torch.pca_lowrank(P, q=D, center=True)
    # Bring to CPU and save relevant stats
    V = V[:, :D].contiguous().cpu()  # (D, D)
    S = S[:D].contiguous().cpu()     # (D,)
    # U = U[:D].contiguous().cpu()  # (M, D)
    # Explained variance ratio from singular values (centered): proportional to S^2
    evr = (S ** 2)
    evr = evr / evr.sum() if float(evr.sum()) > 0 else evr

    out_path = os.path.join(os.environ.get("PCA_DIR", "./data/pca"), args.out or "pca_patch" + f"_{D}_s{step}.pt")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "V": V,  # components as (D, k); here k=D
        "S": S,
        "explained_variance_ratio": evr,
        "patch_size": D,
        "step": step,
        "num_patches": int(P.shape[0]),
    }, out_path)
    print(f"[PCA] Saved PCA basis to {out_path} with V={tuple(V.shape)}")

    if args.plot:
        try:
            import matplotlib.pyplot as plt  # type: ignore
            base = os.path.splitext(out_path)[0]
            # Spectrum plot
            plt.figure()
            plt.plot(S.numpy())
            plt.yscale('log')
            plt.title('PCA singular values')
            plt.tight_layout()
            plt.savefig(base + "_spectrum.png", dpi=150)
            plt.close()
            # Top-10 components
            k = min(10, V.shape[1])
            plt.figure()
            for i in range(k):
                plt.plot(V[:, i].numpy() + i, label=f"PC{i+1}")
            plt.title('Top PCA components (offset)')
            plt.tight_layout()
            plt.savefig(base + "_top10.png", dpi=150)
            plt.close()
            print(f"[PCA] Saved plots next to {out_path}")
        except Exception as e:
            print(f"[PCA] Plotting skipped: {e}")


if __name__ == "__main__":
    main()


# python -m src.prepca.precompute_pca --patch-size 32 --step 32