"""Regression plotter for test-time evaluation - uses shared viz_utils."""

import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List
from scipy import stats

from .viz_utils import (
    denormalize,
    calculate_metrics,
    plot_predictions_vs_true,
    plot_residual_distribution,
    plot_error_vs_true
)


class RegressionPlotter:
    """
    Comprehensive plotter for regression model evaluation.
    Supports multi-output regression (e.g., Teff, log_g, M_H).
    """
    
    def __init__(self, predictions, labels, param_names=None, logger=None, save_dir='./results/test_plots',
                 label_norm=None, label_mean=None, label_std=None, label_min=None, label_max=None):
        """
        Args:
            predictions: np.ndarray of shape (n_samples,) or (n_samples, n_outputs)
            labels: np.ndarray of shape (n_samples,) or (n_samples, n_outputs)
            param_names: List of parameter names (e.g., ['Teff', 'log_g', 'M_H'])
            logger: Optional W&B logger
            save_dir: Directory to save plots
            label_norm: Normalization type ('standard', 'zscore', 'minmax', or None)
            label_mean: Mean values for denormalization
            label_std: Std values for denormalization
            label_min: Min values for denormalization
            label_max: Max values for denormalization
        """
        self.predictions = np.asarray(predictions)
        self.labels = np.asarray(labels)
        self.logger = logger
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Store normalization info for fixed ranges
        self.label_norm = label_norm
        self.label_min_orig = label_min
        self.label_max_orig = label_max
        
        # Determine if multi-output
        self.is_multi_output = len(self.predictions.shape) > 1 and self.predictions.shape[1] > 1
        
        if self.is_multi_output:
            self.n_outputs = self.predictions.shape[1]
            self.param_names = param_names or ['Teff', 'log_g', 'M_H'][:self.n_outputs]
        else:
            self.n_outputs = 1
            self.param_names = param_names or ['Parameter']
            if len(self.predictions.shape) == 1:
                self.predictions = self.predictions.reshape(-1, 1)
                self.labels = self.labels.reshape(-1, 1)
        
        # Denormalize using shared utility
        self.predictions = denormalize(self.predictions, label_norm, label_mean, label_std, label_min, label_max)
        self.labels = denormalize(self.labels, label_norm, label_mean, label_std, label_min, label_max)
        
        if label_norm:
            print(f"[RegressionPlotter] Denormalized with {label_norm}")
        
        # Initialize fixed ranges for consistent plotting with GIF
        self._initialize_fixed_ranges()
    
    def _initialize_fixed_ranges(self):
        """Initialize fixed axis ranges using original parameter ranges (same as VizCallback)."""
        self.fixed_xlim = []
        self.fixed_ylim = []
        
        # Use original parameter ranges if available (minmax normalization)
        if self.label_norm == 'minmax' and self.label_min_orig is not None and self.label_max_orig is not None:
            # Convert to numpy if tensors
            label_min = self.label_min_orig.cpu().numpy() if hasattr(self.label_min_orig, 'cpu') else np.asarray(self.label_min_orig)
            label_max = self.label_max_orig.cpu().numpy() if hasattr(self.label_max_orig, 'cpu') else np.asarray(self.label_max_orig)
            
            for i in range(self.n_outputs):
                min_val = label_min[i] if len(label_min.shape) > 0 else label_min
                max_val = label_max[i] if len(label_max.shape) > 0 else label_max
                self.fixed_xlim.append((min_val, max_val))
                self.fixed_ylim.append((min_val, max_val))
            print(f"[RegressionPlotter] Using original parameter ranges: min={label_min}, max={label_max}")
        else:
            # Fallback: use data ranges with margin
            for i in range(self.n_outputs):
                label_i = self.labels[:, i]
                label_min, label_max = label_i.min(), label_i.max()
                label_range = label_max - label_min
                margin = label_range * 0.05
                self.fixed_xlim.append((label_min - margin, label_max + margin))
                self.fixed_ylim.append((label_min - margin, label_max + margin))
        
        print(f"[RegressionPlotter] Fixed axis ranges: xlim={self.fixed_xlim}, ylim={self.fixed_ylim}")
    
    def _save_and_log(self, fig, name):
        """Save figure locally and log to W&B if available."""
        has_wandb = self.logger and hasattr(self.logger, 'experiment')
        
        # If wandb is enabled, only log to wandb (don't save locally)
        if has_wandb:
            try:
                import wandb
                wandb.log({name: wandb.Image(fig)})
            except Exception as e:
                print(f"Warning: Could not log figure to W&B: {e}")
        else:
            # Only save locally if wandb is not enabled
            filename = name.replace('/', '_').replace('\\', '_') + '.png'
            filepath = os.path.join(self.save_dir, filename)
            try:
                fig.savefig(filepath, dpi=150, bbox_inches='tight')
                print(f"Saved plot: {filepath}")
            except Exception as e:
                print(f"Warning: Could not save figure {filename}: {e}")
        
        return fig
    
    def plot_predictions_vs_true_all(self):
        """Plot scatter plots of predictions vs true values for all outputs."""
        fig, axes = plt.subplots(1, self.n_outputs, figsize=(6*self.n_outputs, 5))
        if self.n_outputs == 1:
            axes = [axes]
        
        for i, (ax, name) in enumerate(zip(axes, self.param_names)):
            xlim = self.fixed_xlim[i] if i < len(self.fixed_xlim) else None
            ylim = self.fixed_ylim[i] if i < len(self.fixed_ylim) else None
            plot_predictions_vs_true(ax, self.predictions[:, i], self.labels[:, i], 
                                    title=name, xlim=xlim, ylim=ylim)
        
        plt.tight_layout()
        return self._save_and_log(fig, 'test/predictions_vs_true')
    
    def plot_residual_distributions_all(self):
        """Plot residual distribution histograms for all outputs."""
        fig, axes = plt.subplots(1, self.n_outputs, figsize=(6*self.n_outputs, 4))
        if self.n_outputs == 1:
            axes = [axes]
        
        for i, (ax, name) in enumerate(zip(axes, self.param_names)):
            xlim = self.fixed_xlim[i] if i < len(self.fixed_xlim) else None
            plot_residual_distribution(ax, self.predictions[:, i], self.labels[:, i], 
                                      title=name, xlim=xlim)
        
        plt.tight_layout()
        return self._save_and_log(fig, 'test/residual_distributions')
    
    def plot_error_vs_true_all(self):
        """Plot error vs true value for all outputs."""
        fig, axes = plt.subplots(1, self.n_outputs, figsize=(6*self.n_outputs, 5))
        if self.n_outputs == 1:
            axes = [axes]
        
        for i, (ax, name) in enumerate(zip(axes, self.param_names)):
            xlim = self.fixed_xlim[i] if i < len(self.fixed_xlim) else None
            plot_error_vs_true(ax, self.predictions[:, i], self.labels[:, i], 
                              title=name, xlim=xlim)
        
        plt.tight_layout()
        return self._save_and_log(fig, 'test/error_vs_true')
    
    def plot_metrics_comparison(self):
        """Plot bar chart comparing metrics across parameters."""
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        metrics = calculate_metrics(self.predictions, self.labels)
        
        mae_vals = [metrics[f'output_{i}']['mae'] for i in range(self.n_outputs)]
        rmse_vals = [metrics[f'output_{i}']['rmse'] for i in range(self.n_outputs)]
        r2_vals = [metrics[f'output_{i}']['r2'] for i in range(self.n_outputs)]
        
        x = np.arange(self.n_outputs)
        width = 0.25
        
        ax.bar(x - width, mae_vals, width, label='MAE', alpha=0.8)
        ax.bar(x, rmse_vals, width, label='RMSE', alpha=0.8)
        ax.bar(x + width, r2_vals, width, label='R²', alpha=0.8)
        
        ax.set_xlabel('Parameters')
        ax.set_ylabel('Metric Value')
        ax.set_title('Performance Metrics per Parameter')
        ax.set_xticks(x)
        ax.set_xticklabels(self.param_names)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        return self._save_and_log(fig, 'test/metrics_per_parameter')
    
    def plot_residual_correlation(self):
        """Plot correlation heatmap of residuals (multi-output only)."""
        if self.n_outputs < 2:
            return None
        
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        
        residuals_all = self.predictions - self.labels
        corr_matrix = np.corrcoef(residuals_all.T)
        
        im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
        
        ax.set_xticks(np.arange(self.n_outputs))
        ax.set_yticks(np.arange(self.n_outputs))
        ax.set_xticklabels(self.param_names)
        ax.set_yticklabels(self.param_names)
        
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Correlation', rotation=270, labelpad=20)
        
        for i in range(self.n_outputs):
            for j in range(self.n_outputs):
                ax.text(j, i, f'{corr_matrix[i, j]:.2f}',
                       ha="center", va="center", color="black")
        
        ax.set_title('Residual Correlation Heatmap')
        plt.tight_layout()
        return self._save_and_log(fig, 'test/residual_correlation')
    
    def plot_qq(self):
        """Plot Q-Q plots for residual normality check."""
        fig, axes = plt.subplots(1, self.n_outputs, figsize=(6*self.n_outputs, 5))
        if self.n_outputs == 1:
            axes = [axes]
        
        for i, (ax, name) in enumerate(zip(axes, self.param_names)):
            residuals = self.predictions[:, i] - self.labels[:, i]
            stats.probplot(residuals, dist="norm", plot=ax)
            ax.set_title(f'{name}: Q-Q Plot')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return self._save_and_log(fig, 'test/qq_plots')
    
    def plot_comprehensive_summary(self):
        """Plot comprehensive 3-row summary figure."""
        from scipy.stats import norm
        
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(3, self.n_outputs, hspace=0.3, wspace=0.3)
        
        for i, name in enumerate(self.param_names):
            pred_i = self.predictions[:, i]
            label_i = self.labels[:, i]
            xlim = self.fixed_xlim[i] if i < len(self.fixed_xlim) else None
            ylim = self.fixed_ylim[i] if i < len(self.fixed_ylim) else None
            
            # Row 1: Scatter plot
            ax1 = fig.add_subplot(gs[0, i])
            plot_predictions_vs_true(ax1, pred_i, label_i, title=name, xlim=xlim, ylim=ylim)
            
            # Row 2: Residual distribution
            ax2 = fig.add_subplot(gs[1, i])
            plot_residual_distribution(ax2, pred_i, label_i, title=name, xlim=xlim)
            
            # Row 3: Error vs true
            ax3 = fig.add_subplot(gs[2, i])
            plot_error_vs_true(ax3, pred_i, label_i, title=name, xlim=xlim)
        
        fig.suptitle('Comprehensive Regression Analysis Summary', fontsize=16, y=0.995)
        plt.tight_layout()
        return self._save_and_log(fig, 'test/comprehensive_summary')
    
    def print_statistics(self):
        """Print detailed statistics to console."""
        metrics = calculate_metrics(self.predictions, self.labels)
        
        print("\n" + "="*80)
        print("DETAILED TEST STATISTICS")
        print("="*80)
        for i, name in enumerate(self.param_names):
            m = metrics[f'output_{i}']
            residuals = self.predictions[:, i] - self.labels[:, i]
            abs_errors = np.abs(residuals)
            
            print(f"\n{name}:")
            print(f"  MAE:     {m['mae']:.6f}")
            print(f"  RMSE:    {m['rmse']:.6f}")
            print(f"  R²:      {m['r2']:.6f}")
            print(f"  Max Err: {abs_errors.max():.6f}")
            print(f"  Median:  {np.median(abs_errors):.6f}")
            print(f"  Std:     {m['std_error']:.6f}")
            print(f"  Error Percentiles:")
            for p in [50, 75, 90, 95, 99]:
                print(f"    {p}%: {np.percentile(abs_errors, p):.6f}")
        print("="*80 + "\n")
    
    def generate_all_plots(self, quick_mode=False):
        """
        Generate all evaluation plots.
        
        Args:
            quick_mode: If True, only generate essential plots
        """
        print("Generating evaluation plots...")
        
        # Essential plots
        self.plot_predictions_vs_true_all()
        self.plot_residual_distributions_all()
        self.plot_metrics_comparison()
        
        if not quick_mode:
            if self.n_outputs > 1:
                self.plot_residual_correlation()
            self.plot_error_vs_true_all()
            self.plot_qq()
            self.plot_comprehensive_summary()
        else:
            self.plot_comprehensive_summary()
        
        self.print_statistics()
        
        has_wandb = self.logger and hasattr(self.logger, 'experiment')
        if has_wandb:
            print("All plots logged to WandB")
        else:
            print(f"All plots saved to: {self.save_dir}")
