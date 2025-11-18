"""Dataset primitives used across spectral models.

This module consolidates all dataset and data-I/O related building blocks
so they can be reused without depending on the broader training utilities
in ``src.basemodule``.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

import h5py
import numpy as np
import pandas as pd
import torch
from scipy import constants
from torch.utils.data import Dataset


class Configurable:
    """Mixin that instantiates objects from nested config dictionaries."""

    init_params: List[str] = []
    config_section: Optional[str] = None

    @classmethod
    def from_config(cls, config: Dict[str, Any]):
        params: Dict[str, Any] = {}
        for base in cls.__mro__[::-1]:
            if issubclass(base, Configurable) and base is not Configurable:
                if base.config_section:
                    section = config.get(base.config_section, {})
                    for param in base.init_params:
                        if param in section:
                            params[param] = section[param]
        return cls(**params)


DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "data"


class BaseDataset(Configurable, Dataset, ABC):
    init_params = [
        "file_path",
        "val_path",
        "test_path",
        "num_samples",
        "num_test_samples",
        "indices",
        "root_dir",
        "param",
        "label_norm",
    ]
    config_section = "data"

    def __init__(
        self,
        file_path: Optional[str] = None,
        num_samples: Optional[int] = None,
        test_path: Optional[str] = None,
        num_test_samples: Optional[int] = None,
        val_path: Optional[str] = None,
        indices: Optional[List[int]] = None,
        root_dir: str = "./results",
        param: Optional[str] = None,
        label_norm: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        print(file_path, num_samples, test_path, num_test_samples, val_path, indices, root_dir)
        self.file_path = file_path
        self.val_path = val_path if val_path is not None else file_path
        self.test_path = test_path if test_path is not None else file_path
        self.num_samples = num_samples if num_samples is not None else 1
        self.num_test_samples = (
            num_test_samples if num_test_samples is not None else min(10000, self.num_samples)
        )
        self.indices = indices if indices is not None else [0, 1]
        self.root_dir = root_dir
        # Optional target parameter(s) to load from HDF; e.g., 'T_eff', 'log_g' or ['T_eff','log_g']
        self.param = param
        # Optional label normalization for regression: 'standard'|'zscore'|'minmax'|None
        self.label_norm = (label_norm or "none").lower() if isinstance(label_norm, str) else "none"
        self.test_data_dict: Dict[str, Any] = {}

    def prepare_data(self) -> None:
        """Called only on 1 GPU: prepare the data for training."""

    @abstractmethod
    def load_data(self, stage: Optional[str] = None) -> None:
        """Load the data from disk. Runs on every process."""

    @abstractmethod
    def __getitem__(self, idx: int):
        raise NotImplementedError

    def __len__(self) -> int:
        return self.num_samples


class MaskMixin(Configurable):
    init_params = ["mask_ratio", "mask_filler", "mask", "lvrg_num", "lvrg_mask"]
    config_section = "mask"

    def __init__(
        self,
        mask_ratio: Optional[float] = None,
        mask_filler: Optional[float] = None,
        mask: Optional[List[int]] = None,
        lvrg_num: Optional[int] = None,
        lvrg_mask=None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.mask_ratio = mask_ratio
        self.mask_filler = mask_filler
        self.mask = mask
        self.lvrg_num = lvrg_num
        self.lvrg_mask = None

    def fill_masked(self, tensor: torch.Tensor, filler: Optional[float] = None) -> torch.Tensor:
        if filler is None:
            return tensor[..., self.mask]
        return tensor.masked_fill(~self.mask, filler)

    def create_quantile_mask(self, tensor: torch.Tensor, ratio: float = 0.9) -> torch.Tensor:
        median = torch.median(tensor, dim=0).values
        print("median", median.mean())
        return median < torch.quantile(median, ratio)

    def create_lvrg_mask(self, wave: torch.Tensor, pdxs) -> np.ndarray:
        wave_len = len(wave)
        mask = np.zeros(wave_len, dtype=bool)
        wdxs = np.digitize(pdxs, wave)
        for wdx in wdxs:
            start, end = max(0, wdx - 25), min(wdx + 25, wave_len)
            mask[start:end] = True
        return mask


class NoiseMixin(Configurable):
    init_params = ["noise_level", "noise_max"]
    config_section = "noise"

    def __init__(self, noise_level: float = 1.0, noise_max: Optional[float] = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.noise_level = noise_level
        self.noise_max = noise_max
        self.noisy: Optional[torch.Tensor] = None

    @staticmethod
    def add_noise(flux: torch.Tensor, error: torch.Tensor, noise_level: float) -> torch.Tensor:
        return flux + torch.randn_like(flux) * error * noise_level

    def clamp_sigma(self, sigma: torch.Tensor) -> torch.Tensor:
        sigma = sigma.clamp(min=1e-6, max=self.noise_max)
        if self.noise_max is None:
            self.noise_max = round(sigma.max().item(), 2)
        self.sigma_rms = torch.sqrt(sigma.pow(2).mean(dim=-1)).mean()
        print("sigma_noise", self.sigma_rms)
        return sigma


class SingleSpectrumNoiseDataset(Dataset):
    def __init__(
        self,
        flux_0: torch.Tensor,
        error_0: torch.Tensor,
        noise_level: float = 1.0,
        repeat: int = 1000,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.repeat = repeat
        self.noise_level = noise_level
        self.L = len(flux_0)
        self.flux_0 = flux_0
        self.error_0 = error_0

        torch.manual_seed(seed)
        noise = torch.randn(repeat, self.L) * self.error_0 * self.noise_level
        self.noisy = self.flux_0 + noise
        print(self.noisy.shape, self.flux_0.shape, self.error_0.shape)

    def __len__(self) -> int:
        return self.repeat

    def __getitem__(self, idx: int):
        return self.noisy[idx], self.flux_0, self.error_0


class BaseSpecDataset(MaskMixin, NoiseMixin, BaseDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.noisy = None  # Pre-generated noisy data for val/test (like blindspot.py)

    def get_path_and_samples(self, stage: Optional[str]):
        if stage in {"fit", "train", None}:
            return self.file_path, self.num_samples
        load_path = self.test_path if stage == "test" else self.val_path
        return load_path, self.num_test_samples

    def replace_nan_with_mean(self, tensor: torch.Tensor) -> torch.Tensor:
        nan_mask = torch.isnan(tensor)
        mean_value = torch.median(tensor[~nan_mask])
        tensor[nan_mask] = mean_value
        return tensor

    def fill_nan_with_nearest(self, tensor: torch.Tensor) -> torch.Tensor:
        if torch.isnan(tensor[:, 0]).any():
            tensor[:, 0] = tensor[:, 1]
        if torch.isnan(tensor[:, -1]).any():
            tensor[:, -1] = tensor[:, -2]
        return tensor

    def load_data(self, stage: Optional[str] = None) -> None:
        load_path, num_samples = self.get_path_and_samples(stage)
        print(f"[{stage or 'train'}] loading data from {load_path}, num_samples={num_samples}")
        
        # Check if file exists
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"[{stage or 'train'}] Data file not found: {load_path}")
        
        with h5py.File(load_path, "r") as f:
            self.wave = torch.tensor(f["spectrumdataset/wave"][()], dtype=torch.float32)
            self.flux = torch.tensor(
                f["dataset/arrays/flux/value"][:num_samples], dtype=torch.float32
            )
            self.error = torch.tensor(
                f["dataset/arrays/error/value"][:num_samples], dtype=torch.float32
            )

        self.flux = self.flux.clip(min=0.0)
        # self.flux = self.flux - self.flux.mean(dim=0, keepdim=True)
        
        self.num_samples = self.flux.shape[0]
        self.num_pixels = len(self.wave)
        if self.error.isnan().any():
            self.error = self.fill_nan_with_nearest(self.error)
        self.snr_no_mask = self.flux.norm(dim=-1) / self.error.norm(dim=-1)
        print(self.flux.shape, self.error.shape, self.wave.shape, self.num_samples, self.num_pixels)
        self._finalize_after_load(stage=stage)

    def load_snr(self, stage: Optional[str] = None, load_df: bool = False) -> None:
        load_path, num_samples = self.get_path_and_samples(stage)
        df = pd.read_hdf(load_path)[:num_samples]
        self.z = df["redshift"].values
        self.rv = self.z * constants.c / 1000
        self.mag = df["mag"].values
        self.snr00 = df["snr"].values
        if load_df:
            self.df = df

    def load_z(self, stage: Optional[str] = None) -> None:
        load_path, num_samples = self.get_path_and_samples(stage)
        df = pd.read_hdf(load_path)[:num_samples]
        self.z = df["redshift"].values
        self.rv = self.z * constants.c / 1000

    def load_params(self, stage: Optional[str] = None, load_df: bool = False) -> None:
        load_path, num_samples = self.get_path_and_samples(stage)
        df = pd.read_hdf(load_path)[:num_samples]

        if isinstance(self.param, str) and len(self.param) > 0:
            param_list = [p.strip() for p in self.param.split(",") if p.strip()]
            if len(param_list) == 1:
                if param_list[0] not in df.columns:
                    raise KeyError(
                        f"Requested param '{param_list[0]}' not found in HDF columns: {list(df.columns)}"
                    )
                self.param_values = df[param_list[0]].values
            else:
                for p in param_list:
                    if p not in df.columns:
                        raise KeyError(
                            f"Requested param '{p}' not found in HDF columns: {list(df.columns)}"
                        )
                self.param_values = df[param_list].values
        elif isinstance(self.param, (list, tuple)) and len(self.param) > 0:
            for p in self.param:
                if p not in df.columns:
                    raise KeyError(
                        f"Requested param '{p}' not found in HDF columns: {list(df.columns)}"
                    )
            self.param_values = df[list(self.param)].values
        else:
            self.teff = df["T_eff"].values
            self.mh = df["M_H"].values
            self.am = df["a_M"].values
            self.cm = df["C_M"].values
            self.logg = df["log_g"].values
            self.logg2 = self.logg < 2.5
        if load_df:
            self.df = df

    def __getitem__(self, idx: int):
        return self.flux[idx], self.error[idx]

    def apply_mask(self) -> None:
        self.flux = self.fill_masked(self.flux, filler=self.mask_filler)
        self.error = self.fill_masked(self.error, filler=self.mask_filler)
        self.wave = self.wave[self.mask] if self.mask is not None else self.wave
        self.num_pixels = len(self.wave)

    def _finalize_after_load(self, stage: Optional[str] = None) -> None:
        # No additional finalization needed
        pass

    def _set_noise(self, seed: int = 42) -> None:
        """Pre-generate noisy data with fixed seed for reproducible validation/test.
        
        This method is called automatically for val/test stages in subclasses.
        Based on blindspot.py implementation for consistent noise generation.
        
        Args:
            seed: Random seed for reproducibility (default: 42)
        """
        if self.noise_level > 0:
            torch.manual_seed(seed)
            noise = torch.randn_like(self.flux) * self.error * self.noise_level
            self.noisy = self.flux + noise
        else:
            self.noisy = None


__all__ = [
    "Configurable",
    "BaseDataset",
    "BaseSpecDataset",
    "MaskMixin",
    "NoiseMixin",
    "SingleSpectrumNoiseDataset",
]
