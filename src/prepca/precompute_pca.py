"""CLI to precompute PCA basis for 1D patch embeddings and save to .pt

This computes PCA over length-`patch_size` windows taken from spectra, then saves
principal components under key 'V' (columns of V are feature-space PCs) so they can initialize the linear patch
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

import torch

from src.prepca.pipeline import compute_pca, load_spectra


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

    data = load_spectra(args.data, dataset_key=args.dataset_key, num_samples=args.limit)
    flux = data["flux"]
    print(f"[PCA] Loaded flux: {tuple(flux.shape)}")

    dev = None
    if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()):
        dev = torch.device("cuda")
    elif args.device == "cpu":
        dev = torch.device("cpu")

    result = compute_pca(
        flux,
        patch_size=args.patch_size,
        step=args.step,
        limit=args.limit,
        device=dev,
    )
    step = int(result["step"].item())
    patch_size = int(result["patch_size"].item())

    out_path = os.path.join(os.environ.get("PCA_DIR", "./data/pca"), args.out or f"pca_patch_{patch_size}_s{step}.pt")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "U": result["scores"],
        "V": result["components"],
        "S": result["singular_values"],
        "mean": result["mean"],
        "explained_variance_ratio": result["explained_variance_ratio"],
        "patch_size": patch_size,
        "step": step,
        "num_patches": int(result["num_patches"].item()),
    }, out_path)
    print(f"[PCA] Saved PCA basis to {out_path} with V={tuple(result['components'].shape)}")

    if args.plot:
        try:
            import matplotlib.pyplot as plt  # type: ignore
            base = os.path.splitext(out_path)[0]
            # Spectrum plot
            plt.figure()
            plt.plot(result["singular_values"].numpy())
            plt.yscale('log')
            plt.title('PCA singular values')
            plt.tight_layout()
            plt.savefig(base + "_spectrum.png", dpi=150)
            plt.close()
            # Top-10 components
            V = result["components"]
            k = min(10, V.shape[1])
            plt.figure()
            for i in range(k):
                plt.plot(V[:, i].numpy() + 0.01* i, label=f"PC{i+1}")
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
