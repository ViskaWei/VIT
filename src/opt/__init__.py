"""
Optimizer hyper-parameter search utilities.

This package provides parallel optimizer sweep functionality across multiple GPUs
to find optimal learning rates and scheduler configurations.

Main components:
- parallel_sweep.py: Multi-GPU parallel hyperparameter search
- optimizer.py: Optimizer and scheduler configuration module
"""

from __future__ import annotations

from .parallel_sweep import ParallelSweepRunner, SweepConfig, SweepResult  # noqa: F401

__all__ = [
    "ParallelSweepRunner",
    "SweepConfig",
    "SweepResult",
]
