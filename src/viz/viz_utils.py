"""Core visualization utilities - shared plotting functions."""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List
import torch


def _format_sample_count(n_samples: int) -> str:
    """Format sample count as 'Xk' or 'X'."""
    if n_samples >= 1000:
        return f"{n_samples/1000:.0f}k"
    return f"{n_samples}"


def format_model_info(model_name: Optional[str] = None,
                     n_samples: Optional[int] = None,
                     n_params: Optional[int] = None,
                     include_epoch: bool = False,
                     epoch: Optional[int] = None,
                     epoch_in_title: bool = False) -> List[str]:
    """
    Format model information into a list of strings.
    
    Args:
        model_name: Name of the model (already includes param count)
        n_samples: Number of training samples
        n_params: Number of model parameters (not used, kept for compatibility)
        include_epoch: Whether to include epoch in the info
        epoch: Current epoch number
        epoch_in_title: If True and epoch provided, always combine with samples
    
    Returns:
        List of formatted info strings [model_name, "50k_Ep10"]
    """
    info_parts = []
    
    if model_name:
        info_parts.append(model_name)
    
    # Build sample+epoch part
    sample_epoch_parts = []
    
    if n_samples is not None:
        sample_epoch_parts.append(_format_sample_count(n_samples))
    
    # Add epoch if requested
    if epoch is not None and (include_epoch or epoch_in_title):
        sample_epoch_parts.append(f"Ep{epoch}")
    
    # Combine with underscore: "50k_Ep10"
    if sample_epoch_parts:
        info_parts.append("_".join(sample_epoch_parts))
    
    return info_parts


# Alias for backward compatibility with title functions
def format_model_info_with_epoch_in_title(model_name: Optional[str] = None,
                                          n_samples: Optional[int] = None,
                                          n_params: Optional[int] = None,
                                          epoch: Optional[int] = None) -> List[str]:
    """Convenience wrapper for when epoch is in title - always combines samples and epoch."""
    return format_model_info(model_name, n_samples, n_params, 
                            include_epoch=False, epoch=epoch, epoch_in_title=True)


def denormalize(data: np.ndarray, norm_type: Optional[str], 
                mean=None, std=None, min_val=None, max_val=None) -> np.ndarray:
    """
    Denormalize data back to original scale.
    
    Args:
        data: Normalized data
        norm_type: 'zscore', 'standard', 'minmax', or None
        mean, std: For zscore/standard normalization
        min_val, max_val: For minmax normalization
    
    Returns:
        Denormalized data
    """
    if norm_type is None or norm_type not in ('standard', 'zscore', 'minmax'):
        return data
    
    eps = 1e-8
    data_denorm = data.copy()
    
    if norm_type in ('standard', 'zscore'):
        if mean is not None and std is not None:
            # Convert to numpy if torch tensors
            mean = mean.cpu().numpy() if hasattr(mean, 'cpu') else np.asarray(mean)
            std = std.cpu().numpy() if hasattr(std, 'cpu') else np.asarray(std)
            
            # Ensure broadcasting works
            if len(mean.shape) > 0 and len(data.shape) > 1:
                mean = mean.flatten()
                std = std.flatten()
            
            # Avoid division by zero
            std = np.where(np.abs(std) < eps, 1.0, std)
            
            # Denormalize: x_original = x_normalized * std + mean
            data_denorm = data * std + mean
    
    elif norm_type == 'minmax':
        if min_val is not None and max_val is not None:
            # Convert to numpy if torch tensors
            min_val = min_val.cpu().numpy() if hasattr(min_val, 'cpu') else np.asarray(min_val)
            max_val = max_val.cpu().numpy() if hasattr(max_val, 'cpu') else np.asarray(max_val)
            
            # Ensure broadcasting works
            if len(min_val.shape) > 0 and len(data.shape) > 1:
                min_val = min_val.flatten()
                max_val = max_val.flatten()
            
            # Avoid division by zero
            denom = max_val - min_val
            denom = np.where(np.abs(denom) < eps, 1.0, denom)
            
            # Denormalize: x_original = x_normalized * (max - min) + min
            data_denorm = data * denom + min_val
    
    return data_denorm


def calculate_metrics(predictions: np.ndarray, labels: np.ndarray) -> dict:
    """
    Calculate regression metrics.
    
    Args:
        predictions: Predicted values, shape (n_samples,) or (n_samples, n_outputs)
        labels: True values, same shape as predictions
    
    Returns:
        Dictionary with MAE, RMSE, R2 for each output
    """
    if len(predictions.shape) == 1:
        predictions = predictions.reshape(-1, 1)
        labels = labels.reshape(-1, 1)
    
    n_outputs = predictions.shape[1]
    metrics = {}
    
    for i in range(n_outputs):
        pred_i = predictions[:, i]
        label_i = labels[:, i]
        residuals = pred_i - label_i
        
        mae = np.mean(np.abs(residuals))
        rmse = np.sqrt(np.mean(residuals**2))
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((label_i - label_i.mean())**2)
        r2 = 1 - (ss_res / (ss_tot + 1e-8))
        
        metrics[f'output_{i}'] = {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'mean_error': np.mean(residuals),
            'std_error': np.std(residuals)
        }
    
    return metrics


def plot_predictions_vs_true(ax, predictions: np.ndarray, labels: np.ndarray, 
                             title: str = '', xlim: Optional[Tuple] = None, 
                             ylim: Optional[Tuple] = None) -> None:
    """
    Plot scatter plot of predictions vs true values.
    
    Args:
        ax: Matplotlib axis
        predictions: Predicted values (n_samples,)
        labels: True values (n_samples,)
        title: Plot title
        xlim: Optional x-axis limits (min, max)
        ylim: Optional y-axis limits (min, max)
    """
    # Scatter plot
    ax.scatter(labels, predictions, alpha=0.5, s=20)
    
    # Perfect prediction line
    if xlim is None:
        min_val, max_val = labels.min(), labels.max()
    else:
        min_val, max_val = xlim
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect')
    
    # Calculate metrics
    residuals = predictions - labels
    mae = np.mean(np.abs(residuals))
    rmse = np.sqrt(np.mean(residuals**2))
    r2 = 1 - np.sum(residuals**2) / (np.sum((labels - labels.mean())**2) + 1e-8)
    
    # Add metrics as text (2 significant figures)
    metrics_text = f'MAE: {mae:.2g}\nRMSE: {rmse:.2g}\nR²: {r2:.2g}'
    ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    ax.set_xlabel('True Value')
    ax.set_ylabel('Predicted Value')
    if title:
        ax.set_title(title)
    
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    
    ax.legend()
    ax.grid(True, alpha=0.3)


def plot_residual_distribution(ax, predictions: np.ndarray, labels: np.ndarray,
                               title: str = '', bins: int = 30, xlim: Optional[Tuple] = None) -> None:
    """
    Plot residual distribution histogram.
    
    Args:
        ax: Matplotlib axis
        predictions: Predicted values (n_samples,)
        labels: True values (n_samples,)
        title: Plot title
        bins: Number of histogram bins
        xlim: Optional x-axis limits - ignored, residuals always centered at 0
    """
    residuals = predictions - labels
    
    # Always create symmetric bins centered at 0
    max_abs = max(abs(np.min(residuals)), abs(np.max(residuals))) * 1.1  # 10% margin
    bins_array = np.linspace(-max_abs, max_abs, bins+1)
    counts, _, _ = ax.hist(residuals, bins=bins_array, alpha=0.7, edgecolor='black', density=True)
    ax.set_xlim(-max_abs, max_abs)
    
    # Zero line and median
    ax.axvline(0, color='r', linestyle='--', lw=2, label='Zero')
    median_err = np.median(residuals)
    ax.axvline(median_err, color='g', linestyle='--', lw=2, 
              label=f'Median={median_err:.2g}')
    
    # Statistics (2 significant figures)
    std_err = np.std(residuals)
    ax.set_xlabel('Residual (Pred - True)')
    ax.set_ylabel('Density')
    if title:
        ax.set_title(f'{title}\nStd={std_err:.2g}')
    else:
        ax.set_title(f'Residual Distribution (Std={std_err:.2g})')
    
    ax.legend()
    ax.grid(True, alpha=0.3)


def plot_error_vs_true(ax, predictions: np.ndarray, labels: np.ndarray,
                      title: str = '', xlim: Optional[Tuple] = None) -> None:
    """
    Plot residual vs true value (detect systematic biases).
    
    Args:
        ax: Matplotlib axis
        predictions: Predicted values (n_samples,)
        labels: True values (n_samples,)
        title: Plot title
        xlim: Optional x-axis limits
    """
    residuals = predictions - labels
    
    # Scatter plot
    ax.scatter(labels, residuals, alpha=0.5, s=20)
    ax.axhline(0, color='r', linestyle='--', lw=2, label='Zero')
    
    # Add standard deviation bands
    std_err = np.std(residuals)
    ax.axhline(std_err, color='orange', linestyle=':', lw=1.5, label=f'±1σ')
    ax.axhline(-std_err, color='orange', linestyle=':', lw=1.5)
    
    ax.set_xlabel('True Value')
    ax.set_ylabel('Residual (Pred - True)')
    if title:
        ax.set_title(title)
    
    if xlim:
        ax.set_xlim(xlim)
    
    ax.legend()
    ax.grid(True, alpha=0.3)


def create_multi_output_figure(predictions: np.ndarray, labels: np.ndarray,
                               param_names: List[str], epoch: int = 0,
                               xlims: Optional[List[Tuple]] = None,
                               ylims: Optional[List[Tuple]] = None,
                               model_name: Optional[str] = None,
                               n_samples: Optional[int] = None,
                               n_params: Optional[int] = None) -> plt.Figure:
    """
    Create a comprehensive figure with prediction, residual, and error plots for all outputs.
    
    Args:
        predictions: Shape (n_samples, n_outputs)
        labels: Shape (n_samples, n_outputs)
        param_names: List of parameter names
        epoch: Current epoch number
        xlims: List of (min, max) tuples for x-axis limits per output
        ylims: List of (min, max) tuples for y-axis limits per output
        model_name: Name of the model
        n_samples: Number of training samples
        n_params: Number of model parameters
    
    Returns:
        Matplotlib figure
    """
    n_outputs = predictions.shape[1] if len(predictions.shape) > 1 else 1
    if n_outputs == 1:
        predictions = predictions.reshape(-1, 1)
        labels = labels.reshape(-1, 1)
    
    # Create figure with 3 rows: scatter, residual, error vs true
    fig, axes = plt.subplots(3, n_outputs, figsize=(6*n_outputs, 12))
    if n_outputs == 1:
        axes = axes.reshape(-1, 1)
    
    for i, name in enumerate(param_names[:n_outputs]):
        pred_i = predictions[:, i]
        label_i = labels[:, i]
        
        xlim = xlims[i] if xlims else None
        ylim = ylims[i] if ylims else None
        
        # Row 1: Prediction vs True (remove epoch from subplot title)
        plot_predictions_vs_true(axes[0, i], pred_i, label_i, 
                                title=f'{name}', 
                                xlim=xlim, ylim=ylim)
        
        # Row 2: Residual Distribution (centered at 0 with fixed scale)
        plot_residual_distribution(axes[1, i], pred_i, label_i, title=name, xlim=xlim)
        
        # Row 3: Error vs True
        plot_error_vs_true(axes[2, i], pred_i, label_i, title=name, xlim=xlim)
    
    # Build title - epoch in model info at the end
    title_parts = ['Prediction Distribution']
    info_parts = format_model_info_with_epoch_in_title(model_name, n_samples, n_params, epoch)
    if info_parts:
        title_parts.append(" | ".join(info_parts))
    
    title = " - ".join(title_parts)
    fig.suptitle(title, fontsize=14, y=0.995)
    
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    return fig
