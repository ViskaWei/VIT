"""Domain-specific datasets built on top of the base spectral dataset primitives."""

from __future__ import annotations

from typing import Any

import torch

from src.dataloader.base import BaseSpecDataset


class ClassSpecDataset(BaseSpecDataset):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    @classmethod
    def from_config(cls, config):
        return super().from_config(config)

    def load_data(self, stage=None):
        super().load_data(stage)
        self.load_params(stage)
        self.labels = (torch.tensor(self.logg > 2.5)).long()
        # Pre-generate noisy data for validation/test with fixed seed (like blindspot.py)
        if stage in ('val', 'test', 'validate'):
            self._set_noise()

    def __getitem__(self, idx):
        flux, error = super().__getitem__(idx)
        # For val/test, return pre-generated noisy; for train, return flux
        if self.noisy is not None:
            return self.noisy[idx], flux, error, self.labels[idx]
        else:
            return flux, error, self.labels[idx]


class RegSpecDataset(BaseSpecDataset):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.label_mean = None
        self.label_std = None
        self.label_min = None
        self.label_max = None

    @classmethod
    def from_config(cls, config):
        return super().from_config(config)

    def load_data(self, stage=None):
        super().load_data(stage)
        self.load_params(stage)
        if not (
            (isinstance(getattr(self, "param", None), str) and len(self.param) > 0)
            or (isinstance(getattr(self, "param", None), (list, tuple)) and len(self.param) > 0)
        ):
            raise ValueError(
                "Regression requires 'data.param' to be set in the config (string, comma-separated string, or list)."
            )
        self.labels = torch.tensor(self.param_values).float()
        self._maybe_normalize_labels(stage)
        # Pre-generate noisy data for validation/test with fixed seed (like blindspot.py)
        if stage in ('val', 'test', 'validate'):
            self._set_noise()

    def __getitem__(self, idx):
        flux, error = super().__getitem__(idx)
        # For val/test, return pre-generated noisy; for train, return flux
        if self.noisy is not None:
            return self.noisy[idx], flux, error, self.labels[idx]
        else:
            return flux, error, self.labels[idx]

    def _maybe_normalize_labels(self, stage=None, kind=None, eps=1e-8):
        kind = getattr(self, "label_norm", "none")
        if kind not in ("standard", "zscore", "minmax"):
            return
        is_train = stage in (None, "fit", "train")
        if kind in ("standard", "zscore"):
            if is_train or (self.label_mean is None or self.label_std is None):
                self.label_mean = self.labels.mean(dim=0, keepdim=False)
                self.label_std = self.labels.std(dim=0, unbiased=False, keepdim=False)
            std = self.label_std.clone() if isinstance(self.label_std, torch.Tensor) else torch.tensor(self.label_std)
            std = torch.where(std.abs() < eps, torch.ones_like(std), std)
            self.labels = (self.labels - self.label_mean) / std
        elif kind == "minmax":
            if is_train or (self.label_min is None or self.label_max is None):
                self.label_min = self.labels.min(dim=0, keepdim=False).values
                self.label_max = self.labels.max(dim=0, keepdim=False).values
            denom = self.label_max - self.label_min
            denom = torch.where(denom.abs() < eps, torch.ones_like(denom), denom)
            self.labels = (self.labels - self.label_min) / denom
        try:
            m = self.label_mean if isinstance(self.label_mean, torch.Tensor) else None
            s = self.label_std if isinstance(self.label_std, torch.Tensor) else None
            mi = self.label_min if isinstance(self.label_min, torch.Tensor) else None
            ma = self.label_max if isinstance(self.label_max, torch.Tensor) else None

            def _fmt(x):
                if x is None:
                    return None
                x = x.detach().cpu().flatten()
                return [round(float(v), 4) for v in x.tolist()[:4]]

            print(
                f"[{stage or 'all'} data] label normalization '{kind}': mean={_fmt(m)}, std={_fmt(s)}, min={_fmt(mi)}, max={_fmt(ma)}"
            )
        except Exception:
            pass


__all__ = ["ClassSpecDataset", "RegSpecDataset"]
