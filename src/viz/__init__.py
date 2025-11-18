"""Visualization package for ViT training."""

from .viz_callback import VizCallback
from .cka_callback import CKACallback
from .plotter import RegressionPlotter
from .callback_factory import create_viz_callbacks
from .viz_utils import (
    denormalize,
    calculate_metrics,
    plot_predictions_vs_true,
    plot_residual_distribution,
    plot_error_vs_true,
    create_multi_output_figure
)
from .gif_maker import (
    save_gif,
    create_distribution_gif_frame,
    create_activation_gif_frame,
    create_attention_gif_frame,
    create_embedding_gif_frame
)

__all__ = [
    'VizCallback',
    'CKACallback',
    'RegressionPlotter',
    'create_viz_callbacks',
    # Utilities
    'denormalize',
    'calculate_metrics',
    'plot_predictions_vs_true',
    'plot_residual_distribution',
    'plot_error_vs_true',
    'create_multi_output_figure',
    # GIF makers
    'save_gif',
    'create_distribution_gif_frame',
    'create_activation_gif_frame',
    'create_attention_gif_frame',
    'create_embedding_gif_frame',
]