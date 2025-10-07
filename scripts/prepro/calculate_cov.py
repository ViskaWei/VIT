"""Calculate and save covariance statistics from training data.

This script loads flux data from an HDF5 file, computes covariance statistics,
and saves them along with a visualization heatmap.

Usage:
    python scripts/calculate_cov.py --file_path <path_to_h5> --output <path_to_save>
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import h5py
import torch

from src.prepca.preprocessor_utils import compute_covariance_stats


def calculate_covariance(
    file_path: str | Path,
    output_path: str | Path,
    num_samples: int | None = None,
) -> None:
    """Calculate covariance statistics from HDF5 dataset.
    
    Parameters
    ----------
    file_path : str | Path
        Path to HDF5 file containing flux data
    output_path : str | Path
        Path where covariance statistics will be saved
    num_samples : int | None
        Number of samples to use. If None, uses all available samples.
    """
    file_path = Path(file_path)
    output_path = Path(output_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path}")
    
    print(f"Loading data from {file_path}")
    
    # Load data from HDF5
    with h5py.File(file_path, 'r') as f:
        # Load wavelength grid (optional, for heatmap labels)
        if 'spectrumdataset/wave' in f:
            wave = torch.tensor(f['spectrumdataset/wave'][()])
        else:
            wave = None
            print("Warning: 'spectrumdataset/wave' not found in HDF5 file")
        
        # Load flux data
        flux_dataset = f['dataset/arrays/flux/value']
        total_samples = flux_dataset.shape[0]
        
        # Determine how many samples to use
        if num_samples is None or num_samples <= 0 or num_samples > total_samples:
            num_samples = total_samples
            flux = torch.tensor(flux_dataset[:])
        else:
            flux = torch.tensor(flux_dataset[:num_samples])
    
    print(f"Loaded {flux.shape[0]} samples with {flux.shape[1]} features")
    
    # Compute and save covariance statistics
    print(f"Computing covariance statistics...")
    stats = compute_covariance_stats(
        data=flux,
        save_path=output_path,
        wave=wave,
        src_path=file_path,  # Save the source data path
    )
    
    print(f"Done! Statistics saved to {output_path}")
    print(f"  - Mean shape: {stats.mean.shape}")
    print(f"  - Covariance shape: {stats.cov.shape}")
    print(f"  - Number of samples: {stats.num_samples}")
    print(f"  - Eigenvalue range: [{stats.eigvals.min():.3e}, {stats.eigvals.max():.3e}]")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calculate covariance statistics from HDF5 training data"
    )
    parser.add_argument(
        "--file_path",
        type=str,
        default=None,
        help="Path to HDF5 file containing flux data (default: ${TRAIN_DIR}/dataset.h5)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for covariance statistics (default: ${PCA_DIR}/cov.pt)",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Number of samples to use (default: use all available)",
    )
    
    args = parser.parse_args()
    
    # Set default paths using environment variables
    if args.file_path is None:
        train_dir = os.environ.get("TRAIN_DIR")
        if train_dir is None:
            raise ValueError(
                "No --file_path provided and TRAIN_DIR environment variable is not set. "
                "Either provide --file_path or set TRAIN_DIR environment variable."
            )
        args.file_path = os.path.join(train_dir, "dataset.h5")
        print(f"Using default file_path: {args.file_path}")
    
    if args.output is None:
        pca_dir = os.environ.get("PCA_DIR")
        if pca_dir is None:
            raise ValueError(
                "No --output provided and PCA_DIR environment variable is not set. "
                "Either provide --output or set PCA_DIR environment variable."
            )
        args.output = os.path.join(pca_dir, "cov.pt")
        print(f"Using default output: {args.output}")
    
    # Expand environment variables in paths
    file_path = Path(os.path.expandvars(args.file_path)).expanduser()
    output_path = Path(os.path.expandvars(args.output)).expanduser()
    
    calculate_covariance(
        file_path=file_path,
        output_path=output_path,
        num_samples=args.num_samples,
    )


if __name__ == "__main__":
    main()
