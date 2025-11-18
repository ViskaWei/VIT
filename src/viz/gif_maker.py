"""GIF creation utilities for training visualization."""

import io
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, FormatStrFormatter
from PIL import Image
from pathlib import Path
from typing import List, Optional
from scipy.stats import entropy

from .viz_utils import (
    create_multi_output_figure, 
    format_model_info, 
    format_model_info_with_epoch_in_title
)


def fig_to_image(fig: plt.Figure) -> Image.Image:
    """Convert matplotlib figure to PIL Image."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img = Image.open(buf).copy()  # Copy to avoid buffer issues
    plt.close(fig)
    buf.close()
    return img


def add_footer_text(fig: plt.Figure, model_name: Optional[str] = None,
                   n_samples: Optional[int] = None, n_params: Optional[int] = None,
                   epoch: Optional[int] = None, bottom_margin: float = 0.03):
    """
    Add model info as footer text to a figure.
    
    Args:
        fig: Matplotlib figure
        model_name: Name of the model
        n_samples: Number of training samples
        n_params: Number of model parameters
        epoch: Current epoch
        bottom_margin: Bottom margin adjustment
    """
    if any([model_name, n_samples, n_params, epoch]):
        info_parts = format_model_info(model_name, n_samples, n_params, 
                                       include_epoch=True, epoch=epoch)
        if info_parts:
            info_text = " | ".join(info_parts)
            fig.text(0.5, 0.01, info_text, ha='center', fontsize=10, 
                    style='italic', color='gray')
            fig.subplots_adjust(bottom=bottom_margin)


def save_gif(frames: List[Image.Image], save_path: Path, duration: int = 500, name: str = ''):
    """
    Save list of PIL Images as animated GIF.
    
    Args:
        frames: List of PIL Images
        save_path: Output path for GIF
        duration: Duration of each frame in milliseconds
        name: Name for logging
    """
    if not frames:
        print(f"  ⚠ No frames to save for {name}")
        return
    
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    frames[0].save(
        save_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0
    )
    print(f"  ✓ Saved {name or save_path.stem} GIF: {save_path} ({len(frames)} frames)")


def create_distribution_gif_frame(predictions: np.ndarray, labels: np.ndarray,
                                  param_names: List[str], epoch: int,
                                  xlims: Optional[List] = None,
                                  ylims: Optional[List] = None,
                                  model_name: Optional[str] = None,
                                  n_samples: Optional[int] = None,
                                  n_params: Optional[int] = None) -> Image.Image:
    """
    Create a single frame for distribution GIF showing predictions, residuals, and errors.
    
    Args:
        predictions: Shape (n_samples, n_outputs)
        labels: Shape (n_samples, n_outputs)
        param_names: List of parameter names
        epoch: Current epoch
        xlims: Fixed x-axis limits per output
        ylims: Fixed y-axis limits per output
        model_name: Name of the model
        n_samples: Number of training samples
        n_params: Number of model parameters
    
    Returns:
        PIL Image
    """
    fig = create_multi_output_figure(predictions, labels, param_names, epoch, xlims, ylims,
                                     model_name, n_samples, n_params)
    
    return fig_to_image(fig)


def create_activation_gif_frame(activation_stats: dict, epoch: int,
                               model_name: Optional[str] = None,
                               n_samples: Optional[int] = None,
                               n_params: Optional[int] = None) -> Image.Image:
    """
    Create activation statistics visualization frame.
    
    Args:
        activation_stats: Dict with layer names as keys, each containing:
            - 'mean': mean activation
            - 'std': std activation
            - 'sparsity': percentage near zero
            - 'zero_rate': percentage exactly zero
            - 'saturation': percentage near saturation
            - 'activations': flattened activations for histogram
        epoch: Current epoch
        model_name: Name of the model
        n_samples: Number of training samples
        n_params: Number of model parameters
    
    Returns:
        PIL Image
    """
    fig = plt.figure(figsize=(16, 12))
    
    layer_names = list(activation_stats.keys())[:8]  # Limit to 8 layers
    n_layers = len(layer_names)
    
    if n_layers == 0:
        plt.close(fig)
        return None
    
    # Plot 1: Mean ± Std over layers
    ax1 = plt.subplot(3, 2, 1)
    means = [activation_stats[name]['mean'] for name in layer_names]
    stds = [activation_stats[name]['std'] for name in layer_names]
    x = range(len(means))
    ax1.plot(x, means, 'o-', label='Mean', linewidth=2)
    ax1.fill_between(x, np.array(means) - np.array(stds), 
                     np.array(means) + np.array(stds), alpha=0.3)
    ax1.set_xlabel('Layer', fontsize=12)
    ax1.set_ylabel('Activation Value', fontsize=12)
    ax1.set_title('Activation Mean ± Std', fontsize=14)
    # Format y-axis to 2 significant figures
    ax1.ticklabel_format(style='sci', axis='y', scilimits=(-2,2))
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.2g}'))
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Sparsity
    ax2 = plt.subplot(3, 2, 2)
    sparsities = [activation_stats[name]['sparsity'] for name in layer_names]
    ax2.bar(x, sparsities, alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Layer', fontsize=12)
    ax2.set_ylabel('Sparsity', fontsize=12)
    ax2.set_title('Activation Sparsity (% near zero)', fontsize=14)
    ax2.set_ylim([0, 1])
    # Format y-axis to 2 significant figures
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.2g}'))
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Zero rate (dead neurons)
    ax3 = plt.subplot(3, 2, 3)
    zero_rates = [activation_stats[name]['zero_rate'] for name in layer_names]
    ax3.bar(x, zero_rates, alpha=0.7, edgecolor='black', color='orange')
    ax3.set_xlabel('Layer', fontsize=12)
    ax3.set_ylabel('Zero Rate', fontsize=12)
    ax3.set_title('Dead Neuron Rate (exact zero)', fontsize=14)
    ax3.set_ylim([0, 1])
    # Format y-axis to 2 significant figures
    ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.2g}'))
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Saturation rate
    ax4 = plt.subplot(3, 2, 4)
    sat_rates = [activation_stats[name]['saturation'] for name in layer_names]
    ax4.bar(x, sat_rates, alpha=0.7, edgecolor='black', color='red')
    ax4.set_xlabel('Layer', fontsize=12)
    ax4.set_ylabel('Saturation Rate', fontsize=12)
    ax4.set_title('Saturation Rate (|x| > 0.9)', fontsize=14)
    ax4.set_ylim([0, 1])
    # Format y-axis to 2 significant figures
    ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.2g}'))
    ax4.grid(True, alpha=0.3)
    
    # Plot 5-6: Histograms for first and last layer
    if len(layer_names) >= 2:
        ax5 = plt.subplot(3, 2, 5)
        first_acts = activation_stats[layer_names[0]]['activations']
        ax5.hist(first_acts, bins=50, alpha=0.7, edgecolor='black')
        ax5.set_xlabel('Activation Value', fontsize=12)
        ax5.set_ylabel('Count', fontsize=12)
        ax5.set_title(f'First Layer ({layer_names[0][:20]})', fontsize=12)
        ax5.grid(True, alpha=0.3)
        
        ax6 = plt.subplot(3, 2, 6)
        last_acts = activation_stats[layer_names[-1]]['activations']
        ax6.hist(last_acts, bins=50, alpha=0.7, edgecolor='black', color='orange')
        ax6.set_xlabel('Activation Value', fontsize=12)
        ax6.set_ylabel('Count', fontsize=12)
        ax6.set_title(f'Last Layer ({layer_names[-1][:20]})', fontsize=12)
        ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    add_footer_text(fig, model_name, n_samples, n_params, epoch, bottom_margin=0.03)
    
    return fig_to_image(fig)


def create_attention_heatmap_gif_frame(attention_stats: dict, epoch: int,
                                       model_name: Optional[str] = None,
                                       n_samples: Optional[int] = None,
                                       n_params: Optional[int] = None) -> Image.Image:
    """
    Create attention heatmap visualization showing each head separately.
    For each layer, show all attention heads (Query vs Key patterns).
    
    Args:
        attention_stats: Dict with layer names as keys, each containing attention_weights
        epoch: Current epoch
        model_name: Name of the model
        n_samples: Number of training samples
        n_params: Number of model parameters
    
    Returns:
        PIL Image showing grid of attention heatmaps (layers x heads)
    """
    layer_names = list(attention_stats.keys())[:4]
    n_layers = len(layer_names)
    
    if n_layers == 0:
        return None
    
    # First, determine the number of heads
    first_attn = attention_stats[layer_names[0]]['attention_weights']
    if hasattr(first_attn, 'cpu'):
        first_attn = first_attn.cpu().numpy()
    elif hasattr(first_attn, 'numpy'):
        first_attn = first_attn.numpy()
    first_attn = np.asarray(first_attn)
    
    # Get number of heads
    if len(first_attn.shape) == 4:  # (batch, heads, seq, seq)
        n_heads = first_attn.shape[1]
    elif len(first_attn.shape) == 3:  # (heads, seq, seq)
        n_heads = first_attn.shape[0]
    else:  # (seq, seq) - single head
        n_heads = 1
    
    # Create grid: 
    # Top row: CLS attention plots (1 row × n_layers cols)
    # Bottom rows: attention heatmaps (n_heads rows × n_layers cols)
    fig = plt.figure(figsize=(4*n_layers, 2 + 4*n_heads))
    
    # Create grid spec with height ratios (CLS row smaller)
    import matplotlib.gridspec as gridspec
    gs = gridspec.GridSpec(n_heads + 1, n_layers, height_ratios=[1] + [4]*n_heads, 
                          hspace=0.3, wspace=0.15)
    
    # First pass: collect CLS attention and normalized attention values
    percentiles_95 = []
    attn_data = []  # Store processed attention data (without CLS)
    cls_data = []   # Store CLS attention patterns
    
    for layer_idx, layer_name in enumerate(layer_names):
        attn = attention_stats[layer_name]['attention_weights']
        
        # Convert to numpy if needed
        if hasattr(attn, 'cpu'):
            attn = attn.cpu().numpy()
        elif hasattr(attn, 'numpy'):
            attn = attn.numpy()
        attn = np.asarray(attn)
        
        # Process based on shape
        if len(attn.shape) == 4:  # (batch, heads, seq, seq)
            attn_by_head = attn.mean(axis=0)  # Average over batch -> (heads, seq, seq)
        elif len(attn.shape) == 3:  # (heads, seq, seq)
            attn_by_head = attn
        else:  # (seq, seq)
            attn_by_head = attn[np.newaxis, :, :]  # -> (1, seq, seq)
        
        layer_attn_abs = []
        layer_cls = []
        for head_idx in range(n_heads):
            attn_head = attn_by_head[head_idx]  # (seq, seq)
            
            # Extract CLS token attention (first row)
            cls_attn = attn_head[0, :]  # (seq,)
            layer_cls.append(cls_attn)
            
            # Remove CLS from attention map (exclude first row and column)
            attn_no_cls = attn_head[1:, 1:]  # (seq-1, seq-1)
            
            # Normalize: subtract uniform distribution and take absolute value
            N = attn_no_cls.shape[-1]
            uniform = 1.0 / N
            attn_norm = attn_no_cls - uniform
            attn_abs = np.abs(attn_norm)
            
            layer_attn_abs.append(attn_abs)
            # Collect 95th percentile per head (excluding CLS)
            percentiles_95.append(np.percentile(attn_abs, 95))
        
        attn_data.append(layer_attn_abs)
        cls_data.append(layer_cls)
    
    # Global vmax: use max of all 95th percentiles (fast and memory-efficient)
    vmax_global = max(percentiles_95)
    
    # Plot CLS attention in top row - show all heads
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown']
    for layer_idx in range(n_layers):
        ax = plt.subplot(gs[0, layer_idx])
        
        # Plot CLS attention for each head
        cls_layer = cls_data[layer_idx]
        for head_idx in range(n_heads):
            cls_head = cls_layer[head_idx]
            color = colors[head_idx % len(colors)]
            ax.plot(cls_head, linewidth=2, color=color, label=f'H{head_idx}', alpha=0.8)
        
        # Calculate ylim based on all heads
        all_max = max(cls_head.max() for cls_head in cls_layer)
        ax.set_ylim([0, max(0.05, all_max * 1.1)])
        ax.grid(True, alpha=0.3)
        
        # Format y-axis to show only 1 decimal place
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        
        # Legend only on first plot
        if layer_idx == 0:
            ax.legend(fontsize=8, loc='upper right')
        
        # Title only on CLS row
        ax.set_title(f'L{layer_idx}', fontsize=12, fontweight='bold')
        
        # Remove x-axis labels for CLS row
        ax.set_xticklabels([])
    
    # Plot attention heatmaps (without CLS)
    for layer_idx, layer_attn_abs in enumerate(attn_data):
        for head_idx in range(n_heads):
            ax = plt.subplot(gs[head_idx + 1, layer_idx])
            attn_abs = layer_attn_abs[head_idx]
            
            # Plot heatmap with shared vmax
            im = ax.imshow(attn_abs, cmap='inferno', aspect='auto', vmin=0, vmax=vmax_global)
            
            # X-axis label only on bottom row (last head)
            if head_idx == n_heads - 1:
                ax.set_xlabel('Key Position', fontsize=10)
            else:
                ax.set_xticklabels([])
            
            # Y-axis label: show H0, H1 on leftmost column
            if layer_idx == 0:
                ax.set_ylabel(f'H{head_idx}\nQuery', fontsize=10)
            else:
                ax.set_yticklabels([])
    
    # Add single colorbar on the right side with more space
    fig.subplots_adjust(right=0.90)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('|Δ| from uniform', fontsize=11)
    
    # Format colorbar to show integer ticks with scientific notation
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-2, 2))  # Use scientific notation for small/large values
    cbar.ax.yaxis.set_major_formatter(formatter)
    
    # Add title and model info
    title_parts = [f'Attention Patterns (Epoch {epoch})']
    if model_name:
        title_parts.append(model_name)
    title = " - ".join(title_parts)
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig_to_image(fig)


def create_attention_gif_frame(attention_stats: dict, epoch: int,
                              model_name: Optional[str] = None,
                              n_samples: Optional[int] = None,
                              n_params: Optional[int] = None) -> Image.Image:
    """
    Create attention diagnostic panel with summary metrics and patterns.
    
    Layout (3x4 grid):
    Row 1: Entropy Trend | Entropy Heatmap | Top-k Mass | (empty)
    Row 2: L0 CLS Attention | L1 CLS | L2 CLS | L3 CLS  
    Row 3: Distance Profile (overlay) | Distance Heatmap | (empty) | (empty)
    
    Args:
        attention_stats: Dict with layer names as keys, each containing:
            - 'entropy': attention entropy values  
            - 'attention_weights': attention weight matrices
        epoch: Current epoch
        model_name: Name of the model
        n_samples: Number of training samples
        n_params: Number of model parameters
    
    Returns:
        PIL Image
    """
    fig = plt.figure(figsize=(18, 12))
    
    layer_names = list(attention_stats.keys())[:4]
    n_layers = len(layer_names)
    
    if n_layers == 0:
        plt.close(fig)
        return None
    
    # ========== Collect Statistics ==========
    layer_entropies = []
    layer_entropies_std = []
    all_head_entropies = []
    layer_top1 = []
    layer_top3 = []
    
    for layer_name in layer_names:
        attn = attention_stats[layer_name]['attention_weights']
        
        # Convert to numpy if needed
        if hasattr(attn, 'cpu'):
            attn = attn.cpu().numpy()
        elif hasattr(attn, 'numpy'):
            attn = attn.numpy()
        attn = np.asarray(attn)
        
        # Compute entropy per head
        if len(attn.shape) == 4:  # (batch, heads, seq, seq)
            attn_avg = attn.mean(axis=0)  # -> (heads, seq, seq)
        elif len(attn.shape) == 3:  # (heads, seq, seq)
            attn_avg = attn
        else:  # (seq, seq)
            attn_avg = attn[np.newaxis, :, :]  # -> (1, seq, seq)
        
        per_head_ent = []
        for h in range(attn_avg.shape[0]):
            entropies_h = [entropy(row + 1e-10) for row in attn_avg[h]]
            per_head_ent.append(np.mean(entropies_h))
        
        layer_entropies.append(np.mean(per_head_ent))
        layer_entropies_std.append(np.std(per_head_ent))
        all_head_entropies.append(per_head_ent)
        
        # Compute top-k mass
        top1_masses = []
        top3_masses = []
        n_heads = attn_avg.shape[0]
        seq_len = attn_avg.shape[1]
        
        for h in range(n_heads):
            for i in range(seq_len):
                row = attn_avg[h, i, :]
                top1_masses.append(np.max(row))
                # Only compute top-3 if sequence length >= 3
                if len(row) >= 3:
                    top3_masses.append(np.partition(row, -3)[-3:].sum())
                else:
                    top3_masses.append(row.sum())
        
        layer_top1.append(np.mean(top1_masses))
        layer_top3.append(np.mean(top3_masses))
    
    # ========== Row 1: Summary Metrics ==========
    x = range(n_layers)
    
    # 1.1 Entropy Trend
    ax1 = plt.subplot(3, 4, 1)
    ax1.errorbar(x, layer_entropies, yerr=layer_entropies_std,
                fmt='o-', capsize=5, linewidth=2, markersize=8, color='tab:blue')
    ax1.set_xlabel('Layer', fontsize=11)
    ax1.set_ylabel('Mean Entropy (nats)', fontsize=11)
    ax1.set_title('Entropy Trend', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'L{i}' for i in x])
    ax1.grid(True, alpha=0.3)
    
    # 1.2 Entropy Heatmap (Layer × Head)
    ax2 = plt.subplot(3, 4, 2)
    max_heads = max(len(h) for h in all_head_entropies)
    entropy_matrix = np.zeros((n_layers, max_heads))
    for i, heads in enumerate(all_head_entropies):
        entropy_matrix[i, :len(heads)] = heads
    
    im = ax2.imshow(entropy_matrix.T, cmap='RdYlGn', aspect='auto')
    ax2.set_xlabel('Layer', fontsize=11)
    ax2.set_ylabel('Head', fontsize=11)
    ax2.set_title('Per-Head Entropy', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'L{i}' for i in x])
    plt.colorbar(im, ax=ax2, label='Entropy', fraction=0.046, pad=0.04)
    
    # 1.3 Top-k Mass
    ax3 = plt.subplot(3, 4, 3)
    ax3.plot(x, layer_top1, 'o-', label='Top-1', linewidth=2, markersize=8, color='tab:orange')
    ax3.plot(x, layer_top3, 's-', label='Top-3', linewidth=2, markersize=8, color='tab:green')
    ax3.set_xlabel('Layer', fontsize=11)
    ax3.set_ylabel('Attention Mass', fontsize=11)
    ax3.set_title('Concentration', fontsize=12, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels([f'L{i}' for i in x])
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([0, 1])
    
    # ========== Row 2: CLS Token Attention ==========
    for idx, layer_name in enumerate(layer_names):
        ax = plt.subplot(3, 4, 5 + idx)
        attn = attention_stats[layer_name]['attention_weights']
        
        # Convert to numpy if needed
        if hasattr(attn, 'cpu'):
            attn = attn.cpu().numpy()
        elif hasattr(attn, 'numpy'):
            attn = attn.numpy()
        attn = np.asarray(attn)
        
        # Extract CLS attention (average over batch and heads)
        if len(attn.shape) == 4:  # (batch, heads, seq, seq)
            cls_attn = attn.mean(axis=(0, 1))[0, :]  # -> (seq,)
        elif len(attn.shape) == 3:  # (heads, seq, seq)
            cls_attn = attn.mean(axis=0)[0, :]  # -> (seq,)
        else:  # (seq, seq)
            cls_attn = attn[0, :]  # -> (seq,)
        
        # Ensure 1D
        if len(cls_attn.shape) != 1:
            print(f"Warning: cls_attn has unexpected shape {cls_attn.shape} for layer {layer_name}")
            cls_attn = cls_attn.flatten()
        
        ax.plot(cls_attn, linewidth=2, color='tab:purple')
        ax.set_xlabel('Key Position', fontsize=10)
        ax.set_ylabel('Weight', fontsize=10)
        ax.set_title(f'L{idx}: CLS Attention', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, max(0.05, cls_attn.max() * 1.1)])
    
    # ========== Row 3: Distance Profile ==========
    
    # 3.1 All layers overlaid
    ax_dist1 = plt.subplot(3, 4, 9)
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
    for idx, layer_name in enumerate(layer_names):
        attn = attention_stats[layer_name]['attention_weights']
        
        # Convert to numpy if needed
        if hasattr(attn, 'cpu'):
            attn = attn.cpu().numpy()
        elif hasattr(attn, 'numpy'):
            attn = attn.numpy()
        attn = np.asarray(attn)
        
        # Average over batch and heads to get (seq, seq)
        if len(attn.shape) == 4:  # (batch, heads, seq, seq)
            attn_avg = attn.mean(axis=(0, 1))
        elif len(attn.shape) == 3:  # (heads, seq, seq)
            attn_avg = attn.mean(axis=0)
        else:  # (seq, seq)
            attn_avg = attn
        
        # Ensure we have 2D array
        if len(attn_avg.shape) != 2:
            print(f"Warning: attn_avg has unexpected shape {attn_avg.shape} for layer {layer_name}")
            continue
        
        # Compute distance profile
        # Use minimum dimension to ensure we don't go out of bounds
        seq_len_query = attn_avg.shape[0]
        seq_len_key = attn_avg.shape[1]
        seq_len = min(seq_len_query, seq_len_key)
        
        max_dist = min(50, seq_len - 1)
        distance_weights = {d: [] for d in range(max_dist + 1)}
        
        for i in range(seq_len):
            for j in range(seq_len):
                d = abs(i - j)
                if d <= max_dist:
                    # Convert to scalar to avoid indexing issues
                    val = float(attn_avg[i, j])
                    distance_weights[d].append(val)
        
        distances = []
        weights = []
        for d in sorted(distance_weights.keys()):
            if distance_weights[d]:
                distances.append(d)
                weights.append(np.mean(distance_weights[d]))
        
        ax_dist1.plot(distances, weights, 'o-', label=f'L{idx}', 
                     alpha=0.7, linewidth=2, color=colors[idx % len(colors)])
    
    ax_dist1.set_xlabel('Distance |i-j|', fontsize=11)
    ax_dist1.set_ylabel('Mean Attention', fontsize=11)
    ax_dist1.set_title('Distance Profile', fontsize=12, fontweight='bold')
    ax_dist1.legend(fontsize=10)
    ax_dist1.grid(True, alpha=0.3)
    ax_dist1.set_yscale('log')
    
    # 3.2 Distance Heatmap
    ax_dist2 = plt.subplot(3, 4, 10)
    max_dist = 50
    dist_matrix = []
    
    for layer_name in layer_names:
        attn = attention_stats[layer_name]['attention_weights']
        
        # Convert to numpy if needed
        if hasattr(attn, 'cpu'):
            attn = attn.cpu().numpy()
        elif hasattr(attn, 'numpy'):
            attn = attn.numpy()
        attn = np.asarray(attn)
        
        # Average over batch and heads to get (seq, seq)
        if len(attn.shape) == 4:  # (batch, heads, seq, seq)
            attn_avg = attn.mean(axis=(0, 1))
        elif len(attn.shape) == 3:  # (heads, seq, seq)
            attn_avg = attn.mean(axis=0)
        else:  # (seq, seq)
            attn_avg = attn
        
        # Ensure we have 2D array
        if len(attn_avg.shape) != 2:
            print(f"Warning: attn_avg has unexpected shape {attn_avg.shape} for layer {layer_name}")
            # Fill with zeros as fallback
            padded_weights = np.zeros(max_dist + 1)
            dist_matrix.append(padded_weights)
            continue
        
        # Use minimum dimension to handle non-square attention matrices
        seq_len_query = attn_avg.shape[0]
        seq_len_key = attn_avg.shape[1]
        seq_len = min(seq_len_query, seq_len_key)
        
        distance_weights = {d: [] for d in range(min(max_dist + 1, seq_len))}
        
        for i in range(seq_len):
            for j in range(seq_len):
                d = abs(i - j)
                if d <= max_dist and d < seq_len:
                    # Convert to scalar to avoid indexing issues
                    val = float(attn_avg[i, j])
                    distance_weights[d].append(val)
        
        weights = [np.mean(distance_weights[d]) if distance_weights[d] else 0 
                  for d in range(min(max_dist + 1, seq_len))]
        
        # Pad to max_dist
        padded_weights = np.zeros(max_dist + 1)
        padded_weights[:len(weights)] = weights
        dist_matrix.append(padded_weights)
    
    im = ax_dist2.imshow(dist_matrix, cmap='inferno', aspect='auto')
    ax_dist2.set_xlabel('Distance', fontsize=11)
    ax_dist2.set_ylabel('Layer', fontsize=11)
    ax_dist2.set_title('Distance Heatmap', fontsize=12, fontweight='bold')
    ax_dist2.set_yticks(range(n_layers))
    ax_dist2.set_yticklabels([f'L{i}' for i in range(n_layers)])
    plt.colorbar(im, ax=ax_dist2, label='Attention', fraction=0.046, pad=0.04)
    
    # Build title
    title_parts = ['Attention Diagnostic Panel']
    info_parts = format_model_info_with_epoch_in_title(model_name, n_samples, n_params, epoch)
    if info_parts:
        title_parts.append(" | ".join(info_parts))
    
    title = " - ".join(title_parts)
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    return fig_to_image(fig)


def create_embedding_gif_frame(embeddings: np.ndarray, labels: np.ndarray,
                               param_names: List[str], epoch: int,
                               method: str = 'umap',
                               model_name: Optional[str] = None,
                               n_samples: Optional[int] = None,
                               n_params: Optional[int] = None) -> Image.Image:
    """
    Create embedding space visualization showing 3 component pairs for each parameter.
    
    Args:
        embeddings: Shape (n_samples, embedding_dim)
        labels: Shape (n_samples, n_outputs)
        param_names: List of parameter names
        epoch: Current epoch
        method: Dimensionality reduction method ('tsne' or 'umap')
        model_name: Name of the model
        n_samples: Number of training samples
        n_params: Number of model parameters
    
    Returns:
        PIL Image with 3 separate rows (one per parameter)
    """
    method = method.lower()
    
    # Try to reduce to 3D using specified method (need 3 components for pairs)
    n_components = 3
    method_name = None
    projected = None
    
    if method == 'umap':
        try:
            from umap import UMAP
            reducer = UMAP(n_components=n_components, random_state=42, n_neighbors=15, min_dist=0.1)
            projected = reducer.fit_transform(embeddings)
            method_name = 'UMAP'
            print(f"[Viz] Using UMAP for embedding visualization")
        except ImportError:
            print("[Warning] umap-learn not available, falling back to t-SNE")
            print("  Install with: pip install umap-learn")
            method = 'tsne'
    
    if method == 'tsne' and projected is None:
        try:
            from sklearn.manifold import TSNE
            tsne = TSNE(n_components=n_components, random_state=42, perplexity=min(30, len(embeddings) // 2))
            projected = tsne.fit_transform(embeddings)
            method_name = 't-SNE'
            print(f"[Viz] Using t-SNE for embedding visualization")
        except ImportError:
            print("[Warning] scikit-learn not available for t-SNE visualization")
            return None
    
    if projected is None:
        print("[Warning] Could not create embedding visualization")
        return None
    
    # Ensure labels is 2D
    if len(labels.shape) == 1:
        labels = labels.reshape(-1, 1)
    
    n_outputs = min(labels.shape[1], 3)  # Limit to 3 parameters
    
    # Define colormaps for each parameter (matching notebook)
    cmaps = ['RdBu', 'viridis', 'BrBG']  # Teff, log_g, M_H
    
    # Create 3xn_outputs subplot grid (n_outputs rows for parameters, 3 columns for pairs)
    pairs = [(0, 1), (1, 2), (0, 2)]
    fig, axes = plt.subplots(n_outputs, 3, figsize=(12, 4*n_outputs), 
                            sharex='col', sharey='col')
    
    # Handle single parameter case
    if n_outputs == 1:
        axes = axes.reshape(1, -1)
    
    for param_idx in range(n_outputs):
        label_values = labels[:, param_idx]
        param_name = param_names[param_idx] if param_idx < len(param_names) else f'Output {param_idx}'
        cmap = cmaps[param_idx] if param_idx < len(cmaps) else 'viridis'
        
        for col_idx, (i, j) in enumerate(pairs):
            ax = axes[param_idx, col_idx]
            sc = ax.scatter(projected[:, i], projected[:, j], s=10, c=label_values, 
                          cmap=cmap, alpha=0.8, edgecolors='none')
            ax.set_xlabel(f'{method_name} Component {i}', fontsize=10)
            ax.set_ylabel(f'{method_name} Component {j}', fontsize=10)
            ax.grid(True, alpha=0.3)
            
            # Add parameter name on first column
            if col_idx == 0:
                ax.text(-0.15, 0.5, param_name, transform=ax.transAxes,
                       fontsize=12, fontweight='bold', rotation=90,
                       verticalalignment='center')
            
            # Add colorbar on last column
            if col_idx == 2:
                cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
                cbar.set_label(param_name, fontsize=10)
    
    # Build title - epoch in model info at the end
    title_parts = [f'{method_name} Embedding Space']
    info_parts = format_model_info_with_epoch_in_title(model_name, n_samples, n_params, epoch)
    if info_parts:
        title_parts.append(" | ".join(info_parts))
    
    title = " - ".join(title_parts)
    fig.suptitle(title, fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.97])  # Leave space for suptitle
    return fig_to_image(fig)


def create_embedding_collinearity_gif_frame(embeddings: np.ndarray, labels: np.ndarray,
                                           param_names: List[str], epoch: int,
                                           model_name: Optional[str] = None,
                                           n_samples: Optional[int] = None,
                                           n_params: Optional[int] = None,
                                           method: str = 'umap',
                                           n_components: int = 3) -> Image.Image:
    """
    Create a 1x3 visualization showing the best collinearity for each parameter.
    For each parameter, find the component pair with highest correlation and show it.
    
    Args:
        embeddings: Embedding vectors (n_samples, embedding_dim)
        labels: Target values (n_samples, n_outputs)
        param_names: Names of output parameters
        epoch: Current training epoch
        model_name: Name of the model
        n_samples: Number of training samples
        n_params: Number of model parameters
        method: Dimensionality reduction method ('umap' or 'tsne')
        n_components: Number of components for reduction
        
    Returns:
        PIL Image of the collinearity analysis
    """
    # Perform dimensionality reduction
    projected = None
    method_name = method.upper()
    
    if method == 'umap':
        try:
            from umap import UMAP
            reducer = UMAP(n_components=n_components, random_state=42, n_neighbors=15, min_dist=0.1)
            projected = reducer.fit_transform(embeddings)
            method_name = 'UMAP'
            print(f"[Viz] Using UMAP for collinearity visualization")
        except ImportError:
            print("[Warning] umap-learn not available, falling back to t-SNE")
            method = 'tsne'
    
    if method == 'tsne' and projected is None:
        try:
            from sklearn.manifold import TSNE
            tsne = TSNE(n_components=n_components, random_state=42, perplexity=min(30, len(embeddings) // 2))
            projected = tsne.fit_transform(embeddings)
            method_name = 't-SNE'
            print(f"[Viz] Using t-SNE for collinearity visualization")
        except ImportError:
            print("[Warning] scikit-learn not available for t-SNE visualization")
            return None
    
    if projected is None:
        print("[Warning] Could not create collinearity visualization")
        return None
    
    # Ensure labels is 2D
    if len(labels.shape) == 1:
        labels = labels.reshape(-1, 1)
    
    # Only show the first parameter
    param_idx = 0
    label_values = labels[:, param_idx]
    param_name = param_names[param_idx] if param_idx < len(param_names) else f'Output {param_idx}'
    
    # Define colormaps for each parameter
    cmaps = ['RdBu', 'viridis', 'BrBG']  # Teff, log_g, M_H
    cmap = cmaps[param_idx] if param_idx < len(cmaps) else 'viridis'
    
    # Create single subplot
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    
    # Calculate correlations for all component pairs
    best_score = -1
    best_pair = (0, 1)
    best_correlations = {}
    
    pairs = [(0, 1), (1, 2), (0, 2)]
    for i, j in pairs:
        # Calculate correlation with each component
        r_i = np.corrcoef(projected[:, i], label_values)[0, 1]
        r_j = np.corrcoef(projected[:, j], label_values)[0, 1]
        
        # Score: max of individual r² values (which component explains more)
        score = max(abs(r_i), abs(r_j))
        
        if score > best_score:
            best_score = score
            best_pair = (i, j)
            best_correlations = {i: r_i, j: r_j}
    
    # Plot the best pair
    i, j = best_pair
    sc = ax.scatter(projected[:, i], projected[:, j], s=10, c=label_values,
                   cmap=cmap, alpha=0.8, edgecolors='none')
    
    # Add correlation info (1 decimal place for clarity)
    r_i = best_correlations[i]
    r_j = best_correlations[j]
    # Show individual correlations without misleading R² sum
    r2_text = f"r(c{i})={r_i:.1f}\nr(c{j})={r_j:.1f}"
    ax.text(0.02, 0.98, r2_text, transform=ax.transAxes, 
           va='top', ha='left', fontsize=10, 
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.set_title(f'param: {param_name}', fontsize=12, fontweight='bold')
    
    # Remove grid and ticks, but keep axes (spines)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')
    
    # Add colorbar without label (param name already in title)
    plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    
    # Build title
    title_parts = [f'{method_name} Embedding Collinearity Analysis']
    info_parts = format_model_info_with_epoch_in_title(model_name, n_samples, n_params, epoch)
    if info_parts:
        title_parts.append(" | ".join(info_parts))
    
    title = " - ".join(title_parts)
    fig.suptitle(title, fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leave space for suptitle
    return fig_to_image(fig)
