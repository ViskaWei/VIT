"""Hardware detection and configuration utilities.

This module provides centralized hardware detection for:
- GPU/CPU/MPS accelerator detection
- Worker count allocation for data loading
- Optimal resource configuration based on environment

All hardware-related logic should live here for maintainability.
"""

import os
import subprocess
from typing import Tuple, Optional
import torch


#region GPU & Accelerator Detection -------------------------------------------

def detect_system_gpus() -> int:
    """Detect total physical GPUs in the system (ignoring CUDA_VISIBLE_DEVICES).
    
    This helps with proper resource allocation when running parallel single-GPU jobs.
    For example, if system has 8 GPUs but CUDA_VISIBLE_DEVICES=3, we still want to
    allocate workers assuming potential 8-way parallelism.
    
    Returns:
        int: Total number of physical GPUs in the system
    """
    try:
        nvidia_smi = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=index', '--format=csv,noheader'],
            stderr=subprocess.DEVNULL
        ).decode('utf-8').strip()
        
        if nvidia_smi:
            return len(nvidia_smi.split('\n'))
    except Exception:
        pass
    
    # Fallback to PyTorch detection
    return torch.cuda.device_count() if torch.cuda.is_available() else 0


def select_accelerator_and_devices(num_gpus: Optional[int] = None) -> Tuple[str, int]:
    """Select optimal accelerator and device count based on availability.
    
    Priority:
    1. CUDA GPUs (if available and requested)
    2. Apple MPS (if available on macOS)
    3. CPU (fallback)
    
    Args:
        num_gpus: Desired number of GPUs (None = auto-detect all available)
        
    Returns:
        Tuple[str, int]: (accelerator_type, device_count)
            accelerator_type: 'gpu', 'mps', or 'cpu'
            device_count: Number of devices to use
            
    Examples:
        >>> select_accelerator_and_devices(2)  # Request 2 GPUs
        ('gpu', 2)
        
        >>> select_accelerator_and_devices(None)  # Auto-detect
        ('gpu', 8)  # On 8-GPU server
        
        >>> select_accelerator_and_devices(None)  # On Mac
        ('mps', 1)
    """
    # If user explicitly requests GPUs
    if num_gpus and num_gpus > 0:
        if torch.cuda.is_available():
            return 'gpu', num_gpus
        if torch.backends.mps.is_available():
            return 'mps', 1  # MPS only supports 1 device
    
    # Auto-detect: prefer CUDA > MPS > CPU
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        return 'gpu', torch.cuda.device_count()
    if torch.backends.mps.is_available():
        return 'mps', 1
    
    return 'cpu', 1


def get_training_strategy(device_count: int) -> str:
    """Determine optimal distributed training strategy.
    
    Args:
        device_count: Number of devices being used
        
    Returns:
        str: PyTorch Lightning strategy ('ddp', 'auto', etc.)
    """
    return 'ddp' if device_count and device_count > 1 else 'auto'

#endregion GPU & Accelerator Detection ----------------------------------------

#region Worker Allocation -----------------------------------------------------

def is_server_environment(cpu_count: int, total_gpus: int) -> bool:
    """Determine if running on a server vs local machine.
    
    Args:
        cpu_count: Number of CPU cores
        total_gpus: Total number of GPUs in system
        
    Returns:
        bool: True if detected as server environment
    """
    return cpu_count >= 32 and total_gpus >= 4


def calculate_optimal_workers(
    cpu_count: int,
    gpu_count: int,
    total_system_gpus: int,
    batch_size: int,
    is_server: bool
) -> int:
    """Calculate optimal number of dataloader workers.
    
    Strategy:
    - Server + Single GPU: Assume parallel execution, divide CPUs fairly
    - Server + Multi GPU (2-4): Moderate workers per GPU
    - Server + Multi GPU (5+): High workers for single large job
    - Local: Use 0 workers (avoid macOS multiprocessing issues)
    
    Args:
        cpu_count: Total CPU cores available
        gpu_count: Number of GPUs requested for this job
        total_system_gpus: Total physical GPUs in the system
        batch_size: Training batch size
        is_server: Whether running on server environment
        
    Returns:
        int: Optimal number of workers
    """
    if not is_server:
        # Local/Mac: use 0 workers to avoid multiprocessing issues
        return 0
    
    # Server environment: optimize based on GPU count
    if gpu_count == 1:
        # Single GPU per job (typical for parallel sweeps/experiments)
        # Assume worst case: user might run jobs on all available GPUs
        # Conservative: divide CPUs by total GPUs to avoid oversubscription
        # Leave 1 CPU per GPU for main process and overhead
        workers_per_job = max(1, (cpu_count - total_system_gpus) // total_system_gpus)
        
        # Cap workers based on I/O intensity and batch size
        # Smaller batches need more workers to keep GPU fed
        if batch_size < 128:
            # Small batch: increase workers for better I/O throughput
            base_workers = min(workers_per_job, 12)
        elif batch_size < 512:
            # Medium batch: balanced
            base_workers = min(workers_per_job, 10)
        else:
            # Large batch: fewer workers (GPU-bound)
            base_workers = min(workers_per_job, 8)
        
        return max(4, min(base_workers, 12))  # Safe range: 4-12
    
    elif gpu_count <= 4:
        # Multi-GPU job (2-4 GPUs): moderate parallelism
        # User likely running 1-2 such jobs at a time
        base_workers = min(6 * gpu_count, 32)
        return min(base_workers, cpu_count - 1, 48)
    
    else:
        # Large multi-GPU job (5+ GPUs): high parallelism
        # User likely running only 1 such job at a time
        if batch_size >= 512:
            base_workers = min(4 * gpu_count, 32)
        elif batch_size >= 128:
            base_workers = min(6 * gpu_count, 48)
        else:
            base_workers = min(8 * gpu_count, 63)
        return min(base_workers, cpu_count - 1, 63)


def auto_detect_num_workers(
    gpu_count: Optional[int] = None,
    batch_size: int = 256,
    verbose: bool = True
) -> int:
    """Automatically detect optimal number of dataloader workers.
    
    Priority order:
    1. Environment variable NUM_WORKERS (highest priority)
    2. Auto-detection based on system resources
    
    Args:
        gpu_count: Number of GPUs for this job (None = auto-detect)
        batch_size: Training batch size
        verbose: Whether to print detection info
        
    Returns:
        int: Optimal number of workers
    """
    # Check environment variable override first
    env_num_workers = os.environ.get('NUM_WORKERS')
    if env_num_workers is not None:
        num_workers = int(env_num_workers)
        if verbose:
            print(f"[WorkerUtils] Using NUM_WORKERS from environment: {num_workers}")
        return num_workers
    
    # Auto-detection
    cpu_count = os.cpu_count() or 1
    if gpu_count is None:
        gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    
    total_system_gpus = detect_system_gpus()
    is_server = is_server_environment(cpu_count, total_system_gpus)
    
    num_workers = calculate_optimal_workers(
        cpu_count=cpu_count,
        gpu_count=gpu_count,
        total_system_gpus=total_system_gpus,
        batch_size=batch_size,
        is_server=is_server
    )
    
    if verbose:
        if is_server:
            print(
                f"[WorkerUtils] Auto-detected SERVER environment: "
                f"{cpu_count} CPUs, {gpu_count} GPU(s) requested "
                f"(system has {total_system_gpus} GPUs), batch_size={batch_size} "
                f"-> using {num_workers} workers"
            )
        else:
            print(
                f"[WorkerUtils] Auto-detected LOCAL environment: "
                f"{cpu_count} CPUs, {gpu_count} GPUs "
                f"-> using {num_workers} workers (single-process)"
            )
    
    return num_workers


def get_num_workers_from_config(
    config: dict,
    verbose: bool = True
) -> Tuple[int, int]:
    """Extract num_workers and batch_size from config with smart defaults.
    
    Args:
        config: Configuration dictionary
        verbose: Whether to print detection info
        
    Returns:
        Tuple[int, int]: (num_workers, batch_size)
    """
    train_config = config.get('train', {})
    
    # Accept both 'num_workers' and legacy 'workers'
    num_workers_config = train_config.get('num_workers', train_config.get('workers', None))
    batch_size = train_config.get('batch_size', 256)
    
    if num_workers_config is not None:
        # User explicitly set in config
        num_workers = int(num_workers_config)
        if verbose:
            print(f"[WorkerUtils] Using num_workers from config: {num_workers}")
        return num_workers, batch_size
    
    # Auto-detect based on system and config
    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    num_workers = auto_detect_num_workers(
        gpu_count=gpu_count,
        batch_size=batch_size,
        verbose=verbose
    )
    
    return num_workers, batch_size

#endregion Worker Allocation --------------------------------------------------


__all__ = [
    # GPU & Accelerator Detection
    'detect_system_gpus',
    'select_accelerator_and_devices',
    'get_training_strategy',
    # Worker Allocation
    'is_server_environment',
    'calculate_optimal_workers',
    'auto_detect_num_workers',
    'get_num_workers_from_config',
]
