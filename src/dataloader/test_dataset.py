"""Synthetic dataset helpers for quick evaluation/integration tests."""

from __future__ import annotations

from typing import Optional

import torch

from src.dataloader.base import BaseSpecDataset
from src.utils import make_dummy_spectra


def _normalize_task(config: Optional[dict]) -> str:
    config = config or {}
    model_cfg = config.get("model", {}) or {}
    task = (model_cfg.get("task_type") or model_cfg.get("task") or "cls").lower()
    if task in ("classification", "cls", "class"):
        return "cls"
    return "reg"


class TestDataset(BaseSpecDataset):
    def __init__(self, task: str = "classification", **kwargs):
        super().__init__(**kwargs)
        self.task = task

    @classmethod
    def from_config(cls, config):
        dataset = super().from_config(config)
        dataset.task = "regression" if _normalize_task(config) == "reg" else "classification"
        return dataset

    def load_data(self, stage=None):
        spectra = make_dummy_spectra(self.num_samples, 4096)
        self.flux = spectra
        self.error = torch.zeros_like(self.flux) + 1e-3
        if self.task == "regression":
            self.labels = torch.randn(spectra.shape[0])
        else:
            self.labels = torch.randint(0, 2, (spectra.shape[0],)).long()
        self.num_samples = self.flux.shape[0]
        self._finalize_after_load(stage)
        # Pre-generate noisy data for validation/test with fixed seed (like blindspot.py)
        if stage in ('val', 'test', 'validate'):
            self._set_noise()

    def __getitem__(self, idx):
        # For val/test, return pre-generated noisy; for train, return flux
        if self.noisy is not None:
            return self.noisy[idx], self.flux[idx], self.error[idx], self.labels[idx]
        else:
            return self.flux[idx], self.error[idx], self.labels[idx]


__all__ = ["TestDataset"]

# Prevent pytest from treating this as a test case during collection.
TestDataset.__test__ = False
