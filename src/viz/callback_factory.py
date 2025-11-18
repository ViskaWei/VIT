"""Factory functions for creating visualization callbacks."""

from .viz_callback import VizCallback
from .cka_callback import CKACallback


def create_viz_callbacks(viz_config, save_enabled=True):
    """
    Create visualization callbacks from config.
    
    Args:
        viz_config: Dictionary with visualization settings
        save_enabled: Whether local saving is enabled (from train.save)
        
    Returns:
        List of callback instances
    """
    callbacks = []
    
    if not viz_config.get('enable', False):
        return callbacks
    
    # Determine logging frequency (steps take priority over epochs)
    log_every_n_steps = viz_config.get('log_every_n_steps', None)
    log_every_n_epochs = viz_config.get('log_every_n_epochs', 10)
    
    # Respect train.save setting: only save locally if save_enabled
    save_local = viz_config.get('save_local', save_enabled)
    
    # Main visualization callback (memory-optimized)
    viz_callback = VizCallback(
        save_dir=viz_config.get('save_dir', './results/viz'),
        log_every_n_epochs=log_every_n_epochs,
        log_every_n_steps=log_every_n_steps,
        num_viz_samples=viz_config.get('num_viz_samples', 256),
        max_batches_to_track=viz_config.get('max_batches_to_track', 5),
        create_distribution_gif=viz_config.get('create_distribution_gif', True),
        create_activation_gif=viz_config.get('create_activation_gif', False),
        create_attention_gif=viz_config.get('create_attention_gif', True),
        create_embedding_gif=viz_config.get('create_embedding_gif', False),
        create_collinearity_gif=viz_config.get('create_collinearity_gif', False),
        embedding_method=viz_config.get('embedding_method', 'umap'),
        gif_duration=viz_config.get('gif_duration', 500),
        activation_layer_names=viz_config.get('activation_layer_names', None),
        sample_rate=viz_config.get('sample_rate', 0.15),
        save_local=save_local,  # NEW: control local saving
    )
    callbacks.append(viz_callback)
    
    # Print logging configuration
    save_mode = "saved locally + uploaded" if save_local else "uploaded to WandB only"
    if log_every_n_steps:
        print(f'[VizFactory] Visualization enabled: every {log_every_n_steps} steps ({save_mode})')
    else:
        print(f'[VizFactory] Visualization enabled: every {log_every_n_epochs} epochs ({save_mode})')
    
    # Print enabled visualizations
    enabled_viz = []
    if viz_config.get('create_distribution_gif', True):
        enabled_viz.append('distribution')
    if viz_config.get('create_activation_gif', False):
        enabled_viz.append('activation')
    if viz_config.get('create_attention_gif', True):
        enabled_viz.append('attention')
    if viz_config.get('create_embedding_gif', False):
        enabled_viz.append(f'embedding({viz_config.get("embedding_method", "umap")})')
    if viz_config.get('create_collinearity_gif', False):
        enabled_viz.append('collinearity')
    
    if enabled_viz:
        print(f'[VizFactory] Enabled: {", ".join(enabled_viz)}')
    
    # CKA analysis callback (separate)
    if viz_config.get('compute_cka', True):
        cka_callback = CKACallback(
            save_dir=viz_config.get('save_dir', './results/viz'),
            log_every_n_epochs=log_every_n_epochs,
            log_every_n_steps=log_every_n_steps,
            cka_layers=viz_config.get('cka_layers', None),
        )
        callbacks.append(cka_callback)
        print(f'[VizFactory] CKA analysis enabled (monitors layer learning)')
    
    return callbacks
