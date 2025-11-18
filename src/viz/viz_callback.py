"""Memory-efficient visualization callback - optimized version."""

import numpy as np
import torch
import lightning as L
from pathlib import Path
from typing import Optional, List
from collections import defaultdict, deque
from scipy.stats import entropy

from .viz_utils import denormalize
from .gif_maker import (
    create_distribution_gif_frame,
    create_activation_gif_frame,
    create_attention_gif_frame,
    create_attention_heatmap_gif_frame,
    create_embedding_gif_frame,
    create_embedding_collinearity_gif_frame,
    save_gif
)


class VizCallback(L.Callback):
    """
    Memory-efficient visualization callback for network diagnostics.
    
    Features:
    1. Limited batch accumulation with automatic memory management
    2. Sampled activations/attention to reduce memory footprint
    3. Efficient embedding extraction without full hidden states
    4. Immediate cleanup after each epoch
    """
    
    def __init__(
        self,
        save_dir: str = './results/viz',
        log_every_n_epochs: int = 10,
        log_every_n_steps: Optional[int] = None,  # NEW: step-based logging
        num_viz_samples: int = 500,
        max_batches_to_track: int = 10,  # NEW: limit batches tracked
        create_distribution_gif: bool = True,
        create_activation_gif: bool = False,  # Disabled by default
        create_attention_gif: bool = True,
        create_embedding_gif: bool = False,  # Disabled by default
        create_collinearity_gif: bool = False,  # NEW: Collinearity analysis
        embedding_method: str = 'umap',  # 'tsne' or 'umap'
        gif_duration: int = 500,
        activation_layer_names: Optional[List[str]] = None,
        sample_rate: float = 0.1,  # NEW: sample 10% of activations
        save_local: bool = True,  # NEW: control local saving
    ):
        super().__init__()
        self.save_local = save_local
        self.save_dir = Path(save_dir)
        if self.save_local:
            self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_every_n_epochs = log_every_n_epochs
        self.log_every_n_steps = log_every_n_steps  # If set, overrides epoch-based
        self.num_viz_samples = num_viz_samples
        self.max_batches_to_track = max_batches_to_track
        self.create_distribution_gif = create_distribution_gif
        self.create_activation_gif = create_activation_gif
        self.create_attention_gif = create_attention_gif
        self.create_embedding_gif = create_embedding_gif
        self.create_collinearity_gif = create_collinearity_gif
        self.embedding_method = embedding_method.lower()
        self.gif_duration = gif_duration
        self.activation_layer_names = activation_layer_names
        self.sample_rate = sample_rate
        
        # Storage for frames - use deque for automatic memory management
        self.distribution_frames = deque(maxlen=100)  # Keep last 100 frames
        self.activation_frames = deque(maxlen=100)
        self.attention_frames = deque(maxlen=100)
        self.attention_heatmap_frames = deque(maxlen=100)  # NEW: simple heatmaps
        self.embedding_frames = deque(maxlen=100)  # Add embedding frames
        self.collinearity_frames = deque(maxlen=100)  # Add collinearity frames
        
        # Hook storage - use deque to limit size
        self.activation_hooks = {}
        self.attention_hooks = {}
        self.hooks_registered = False
        self.batch_counter = 0
        
        # Normalization parameters
        self.label_norm = None
        self.label_mean = None
        self.label_std = None
        self.label_min = None
        self.label_max = None
        
        # Fixed axis ranges
        self.fixed_xlim = None
        self.fixed_ylim = None
        
    def on_validation_start(self, trainer, pl_module):
        """Register hooks at validation start."""
        # Check if we should log this epoch/step
        if not self._should_log(trainer):
            return
        
        # Get normalization params
        self._get_normalization_params(trainer)
        
        # Register hooks for this validation epoch
        if self.create_activation_gif or self.create_attention_gif:
            # Force model to output attention weights during validation
            if self.create_attention_gif and hasattr(pl_module, 'model'):
                self._enable_attention_output(pl_module.model)
            
            self._register_hooks(pl_module)
    
    def _enable_attention_output(self, model):
        """Monkey-patch model to always output attention weights."""
        try:
            from transformers.models.vit.modeling_vit import ViTSelfAttention
            
            # Patch all ViTSelfAttention modules to always return attention
            for name, module in model.named_modules():
                # Check if this is specifically a ViTSelfAttention instance
                if isinstance(module, ViTSelfAttention):
                    # Store original forward
                    if not hasattr(module, '_original_forward'):
                        module._original_forward = module.forward
                        
                        # Create patched forward that forces output_attentions=True
                        def make_patched_forward(original_forward):
                            def patched_forward(hidden_states, head_mask=None, output_attentions=False):
                                # Always set output_attentions=True
                                return original_forward(hidden_states, head_mask, output_attentions=True)
                            return patched_forward
                        
                        module.forward = make_patched_forward(module._original_forward)
        except Exception as e:
            print(f"[VizCallback] Failed to patch attention output: {e}")
    
    def _should_log(self, trainer):
        """Check if we should log at this epoch/step."""
        # Step-based logging: check if we're close to a logging step
        # Since validation happens at epoch boundaries, check if the current epoch
        # corresponds to a step multiple
        if self.log_every_n_steps is not None:
            # Calculate steps per epoch (approximate)
            steps_per_epoch = trainer.num_training_batches
            current_step = trainer.global_step
            
            # Check if we just passed a logging milestone
            # Allow validation at the end of any epoch that contains a log step
            epoch_start_step = trainer.current_epoch * steps_per_epoch
            epoch_end_step = epoch_start_step + steps_per_epoch
            
            # Log if this epoch contains any multiple of log_every_n_steps
            next_log_step = ((epoch_start_step // self.log_every_n_steps) + 1) * self.log_every_n_steps
            return next_log_step <= epoch_end_step
        
        # Fall back to epoch-based
        return trainer.current_epoch % self.log_every_n_epochs == 0
    
    def _register_hooks(self, pl_module):
        """Register forward hooks with sampling."""
        def activation_hook(name):
            def hook(module, input, output):
                # Only track for limited number of batches
                if self.batch_counter >= self.max_batches_to_track:
                    return
                
                if isinstance(output, tuple):
                    output = output[0]
                
                # Sample activations to reduce memory
                output_flat = output.detach().flatten()
                sample_size = max(1, int(len(output_flat) * self.sample_rate))
                indices = torch.randperm(len(output_flat))[:sample_size]
                sampled = output_flat[indices].cpu()
                
                # Use deque to auto-limit size
                if name not in self.activation_hooks:
                    self.activation_hooks[name] = deque(maxlen=self.max_batches_to_track)
                self.activation_hooks[name].append(sampled)
            return hook
        
        def attention_hook(name):
            def hook(module, input, output):
                # Only track for limited number of batches
                if self.batch_counter >= self.max_batches_to_track:
                    return
                
                # Extract attention weights
                # HuggingFace ViTAttention returns (hidden_states, attention_probs)
                attn_weights = None
                
                if isinstance(output, tuple) and len(output) >= 2:
                    # Second element should be attention_probs
                    candidate = output[1] if output[1] is not None else output[0]
                    
                    if candidate is not None and hasattr(candidate, 'detach') and hasattr(candidate, 'dim'):
                        dims = candidate.dim()
                        shape = candidate.shape
                        # Attention probs should be 4D: (batch, heads, seq, seq) with square matrix
                        if dims == 4 and shape[-2] == shape[-1]:
                            attn_weights = candidate
                        # Or 3D: (batch, seq, seq) if already averaged over heads
                        elif dims == 3 and shape[-2] == shape[-1]:
                            attn_weights = candidate
                
                if attn_weights is not None:
                    # Take mean over batch to reduce memory
                    attn_mean = attn_weights.detach().mean(dim=0).cpu()
                    
                    if name not in self.attention_hooks:
                        self.attention_hooks[name] = deque(maxlen=self.max_batches_to_track)
                    self.attention_hooks[name].append(attn_mean)
            return hook
        
        # Only register hooks for selected layers (limit to 4 layers max)
        activation_count = 0
        attention_count = 0
        
        for name, module in pl_module.named_modules():
            # Activation hooks - limit to 4 layers
            if activation_count < 4 and self.create_activation_gif:
                if 'linear' in name.lower() or 'mlp' in name.lower():
                    if self.activation_layer_names is None or name in self.activation_layer_names:
                        module.register_forward_hook(activation_hook(name))
                        activation_count += 1
            
            # Attention hooks - limit to 4 layers
            # Hook ViTSelfAttention (the inner module that computes attention)
            # rather than ViTAttention wrapper (which doesn't return attention_probs by default)
            if attention_count < 4 and self.create_attention_gif:
                # Target: 'vit.encoder.layer.X.attention.attention' (ViTSelfAttention)
                if 'encoder.layer' in name and name.endswith('.attention.attention'):
                    module.register_forward_hook(attention_hook(name))
                    attention_count += 1
    
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Track batch counter."""
        if self._should_log(trainer):
            self.batch_counter += 1
    
    def on_validation_epoch_end(self, trainer, pl_module):
        """Create and save visualizations."""
        if trainer.sanity_checking:
            return
        
        # Check if we should log
        if not self._should_log(trainer):
            return
        
        current_epoch = trainer.current_epoch
        
        # Get normalization parameters
        self._get_normalization_params(trainer)
        
        print(f"\n[VizCallback] Creating visualizations for epoch {current_epoch}")
        
        # Collect predictions and labels (with memory limit)
        preds_list, labels_list = [], []
        samples_collected = 0
        
        pl_module.eval()
        with torch.no_grad():
            dataloader = trainer.val_dataloaders
            for batch_idx, batch in enumerate(dataloader):
                if samples_collected >= self.num_viz_samples:
                    break
                
                # Handle different batch formats
                if isinstance(batch, (tuple, list)):
                    if len(batch) == 4:
                        inputs, labels = batch[0], batch[3]
                    elif len(batch) == 3:
                        inputs, labels = batch[0], batch[2]
                    else:
                        inputs, labels = batch[0], batch[1]
                elif isinstance(batch, dict):
                    inputs = batch.get('pixel_values', batch.get('flux'))
                    labels = batch['labels']
                else:
                    continue
                
                inputs = inputs.to(pl_module.device)
                
                # Forward pass
                if hasattr(pl_module, 'model'):
                    outputs = pl_module.model(inputs)
                else:
                    outputs = pl_module(inputs)
                
                preds = outputs.logits if hasattr(outputs, 'logits') else outputs
                preds_list.append(preds.cpu())
                labels_list.append(labels.cpu())
                
                samples_collected += len(labels)
                
                # Clear some GPU memory
                del outputs, preds
                if batch_idx % 5 == 0:
                    torch.cuda.empty_cache()
        
        preds = torch.cat(preds_list, dim=0)[:self.num_viz_samples]
        labels = torch.cat(labels_list, dim=0)[:self.num_viz_samples]
        
        # Clear lists to free memory
        del preds_list, labels_list
        
        # Denormalize
        preds_np = self._denormalize(preds.numpy())
        labels_np = self._denormalize(labels.numpy())
        
        # Clear tensors
        del preds, labels
        
        # Determine parameter names
        num_outputs = preds_np.shape[1] if len(preds_np.shape) > 1 else 1
        param_names = self._get_param_names(trainer, num_outputs)
        
        # Initialize fixed axis ranges on first epoch
        if self.fixed_xlim is None and self.fixed_ylim is None:
            self._initialize_fixed_ranges(labels_np, num_outputs)
        
        # Get model information for visualization
        model_name, n_samples, n_params = self._get_model_info(pl_module, trainer)
        
        # Create distribution GIF frame
        if self.create_distribution_gif:
            frame = create_distribution_gif_frame(
                preds_np, labels_np, param_names, current_epoch,
                self.fixed_xlim, self.fixed_ylim,
                model_name, n_samples, n_params
            )
            if frame:
                self.distribution_frames.append(frame)
        
        # Create activation GIF frame
        if self.create_activation_gif and self.activation_hooks:
            activation_stats = self._compute_activation_stats()
            frame = create_activation_gif_frame(activation_stats, current_epoch,
                                               model_name, n_samples, n_params)
            if frame:
                self.activation_frames.append(frame)
        
        # Create attention GIF frame
        if self.create_attention_gif and self.attention_hooks:
            attention_stats = self._compute_attention_stats()
            frame = create_attention_gif_frame(attention_stats, current_epoch,
                                              model_name, n_samples, n_params)
            if frame:
                self.attention_frames.append(frame)
            
            # Also create simple heatmap visualization
            heatmap_frame = create_attention_heatmap_gif_frame(attention_stats, current_epoch,
                                                               model_name, n_samples, n_params)
            if heatmap_frame:
                self.attention_heatmap_frames.append(heatmap_frame)
        
        # Create embedding GIF frame (memory-efficient version)
        embeddings = None
        if self.create_embedding_gif:
            embeddings = self._extract_embeddings_lite(pl_module, trainer.val_dataloaders)
            if embeddings is not None:
                frame = create_embedding_gif_frame(embeddings, labels_np, param_names, 
                                                  current_epoch, method=self.embedding_method,
                                                  model_name=model_name, n_samples=n_samples,
                                                  n_params=n_params)
                if frame:
                    self.embedding_frames.append(frame)
        
        # Create collinearity GIF frame
        if self.create_collinearity_gif:
            # Reuse embeddings from above if available, otherwise extract them
            if embeddings is None:
                embeddings = self._extract_embeddings_lite(pl_module, trainer.val_dataloaders)
            if embeddings is not None:
                frame = create_embedding_collinearity_gif_frame(embeddings, labels_np, param_names,
                                                               current_epoch, method=self.embedding_method,
                                                               model_name=model_name, n_samples=n_samples,
                                                               n_params=n_params)
                if frame:
                    self.collinearity_frames.append(frame)
        
        # Clear hooks immediately
        self._clear_hooks()
        
        # Force garbage collection
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        
        # Log to WandB if available
        self._log_to_wandb(trainer.logger, current_epoch)
    
    def _get_normalization_params(self, trainer):
        """Retrieve normalization parameters from datamodule."""
        try:
            if hasattr(trainer, 'datamodule') and hasattr(trainer.datamodule, 'val'):
                val_dataset = trainer.datamodule.val
                self.label_norm = getattr(val_dataset, 'label_norm', None)
                if self.label_norm in ('standard', 'zscore'):
                    self.label_mean = getattr(val_dataset, 'label_mean', None)
                    self.label_std = getattr(val_dataset, 'label_std', None)
                elif self.label_norm == 'minmax':
                    self.label_min = getattr(val_dataset, 'label_min', None)
                    self.label_max = getattr(val_dataset, 'label_max', None)
        except Exception:
            pass
    
    def _denormalize(self, data: np.ndarray) -> np.ndarray:
        """Denormalize data using stored parameters."""
        return denormalize(data, self.label_norm, 
                          self.label_mean, self.label_std,
                          self.label_min, self.label_max)
    
    def _get_model_info(self, pl_module, trainer) -> tuple:
        """Extract model information for visualization."""
        # Get model name
        model_name = (
            getattr(pl_module.model, 'name', None) if hasattr(pl_module, 'model')
            else getattr(pl_module, 'name', pl_module.__class__.__name__)
        )
        
        # Get number of training samples
        n_samples = None
        try:
            dm = trainer.datamodule
            if hasattr(dm, 'train_dataloader'):
                n_samples = len(dm.train_dataloader().dataset)
            elif hasattr(dm, 'train'):
                n_samples = len(dm.train)
        except:
            pass
        
        # Get number of parameters
        n_params = None
        try:
            model = pl_module.model if hasattr(pl_module, 'model') else pl_module
            n_params = sum(p.numel() for p in model.parameters())
        except:
            pass
        
        # Append parameter count to model name
        if model_name and n_params:
            model_name = f"{model_name}_{n_params/1e6:.2f}M"
        
        return model_name, n_samples, n_params
    
    def _get_param_names(self, trainer, num_outputs: int) -> List[str]:
        """Get parameter names from datamodule or use defaults."""
        try:
            return trainer.datamodule.param_names
        except:
            # Get param from config data.param
            param = trainer.datamodule.config.get('data', {}).get('param')
            if param:
                param_list = [p.strip() for p in param.split(",")] if isinstance(param, str) else list(param)
                return param_list[:num_outputs] if len(param_list) >= num_outputs else param_list
            
            # Fallback to defaults
            default_names = ['Teff', 'log_g', 'M_H']
            return default_names[:num_outputs] if num_outputs <= 3 else [f'Output_{i}' for i in range(num_outputs)]
    
    def _initialize_fixed_ranges(self, labels_np: np.ndarray, num_outputs: int):
        """Initialize fixed axis ranges using original parameter ranges."""
        self.fixed_xlim, self.fixed_ylim = [], []
        
        if self.label_norm == 'minmax' and self.label_min is not None and self.label_max is not None:
            label_min = self.label_min.cpu().numpy() if hasattr(self.label_min, 'cpu') else np.asarray(self.label_min)
            label_max = self.label_max.cpu().numpy() if hasattr(self.label_max, 'cpu') else np.asarray(self.label_max)
            
            for i in range(num_outputs):
                min_val = label_min[i] if len(label_min.shape) > 0 else label_min
                max_val = label_max[i] if len(label_max.shape) > 0 else label_max
                self.fixed_xlim.append((min_val, max_val))
                self.fixed_ylim.append((min_val, max_val))
        else:
            for i in range(num_outputs):
                label_i = labels_np[:, i] if num_outputs > 1 else labels_np
                label_min, label_max = label_i.min(), label_i.max()
                margin = (label_max - label_min) * 0.05
                range_tuple = (label_min - margin, label_max + margin)
                self.fixed_xlim.append(range_tuple)
                self.fixed_ylim.append(range_tuple)
    
    def _compute_activation_stats(self) -> dict:
        """Compute statistics from sampled activations."""
        stats = {}
        for layer_name, activations_deque in list(self.activation_hooks.items())[:4]:
            if not activations_deque:
                continue
            
            # Concatenate sampled activations
            activations = torch.cat(list(activations_deque))
            activations_np = activations.numpy()
            
            stats[layer_name] = {
                'mean': np.mean(activations_np),
                'std': np.std(activations_np),
                'sparsity': np.mean(np.abs(activations_np) < 0.01),
                'zero_rate': np.mean(activations_np == 0),
                'saturation': np.mean(np.abs(activations_np) > 0.9),
                'activations': activations_np
            }
        return stats
    
    def _compute_attention_stats(self) -> dict:
        """Compute statistics from attention weights."""
        stats = {}
        for layer_name, attention_deque in list(self.attention_hooks.items())[:4]:
            if not attention_deque:
                continue
            
            # Average collected attention weights
            attn_list = list(attention_deque)
            attn_mean = torch.stack(attn_list).mean(dim=0)
            attn_np = attn_mean.numpy()
            
            # Compute entropy
            if len(attn_np.shape) >= 2:
                if len(attn_np.shape) == 3:  # (heads, seq, seq)
                    attn_for_entropy = attn_np.mean(axis=0)
                else:
                    attn_for_entropy = attn_np
                
                entropies = [entropy(row + 1e-10) for row in attn_for_entropy]
                mean_entropy = np.mean(entropies)
            else:
                mean_entropy = 0.0
            
            stats[layer_name] = {
                'entropy': mean_entropy,
                'attention_weights': attn_np
            }
        return stats
    
    def _clear_hooks(self):
        """Clear all hook data immediately."""
        for key in list(self.activation_hooks.keys()):
            self.activation_hooks[key].clear()
        for key in list(self.attention_hooks.keys()):
            self.attention_hooks[key].clear()
    
    def _extract_embeddings_lite(self, pl_module, dataloader) -> Optional[np.ndarray]:
        """Extract embeddings efficiently with memory limits."""
        try:
            embeddings_list = []
            samples_collected = 0
            
            pl_module.eval()
            with torch.no_grad():
                for batch_idx, batch in enumerate(dataloader):
                    if samples_collected >= self.num_viz_samples:
                        break
                    
                    # Handle different batch formats
                    if isinstance(batch, (tuple, list)):
                        if len(batch) >= 3:
                            inputs = batch[0]
                        else:
                            inputs = batch[0]
                    elif isinstance(batch, dict):
                        inputs = batch.get('pixel_values', batch.get('flux'))
                    else:
                        continue
                    
                    inputs = inputs.to(pl_module.device)
                    
                    # Try to get embeddings without full hidden states
                    try:
                        # Option 1: Try to get last layer output before classifier
                        if hasattr(pl_module, 'model'):
                            model = pl_module.model
                            # For ViT models, get the output before the final classifier
                            if hasattr(model, 'vit'):
                                # Get transformer output
                                outputs = model.vit(inputs, output_hidden_states=False)
                                if hasattr(outputs, 'last_hidden_state'):
                                    # Use CLS token (first token)
                                    embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                                elif hasattr(outputs, 'pooler_output'):
                                    embeddings = outputs.pooler_output.cpu().numpy()
                                else:
                                    # Fallback: use mean pooling
                                    embeddings = outputs[0][:, 0, :].cpu().numpy()
                            else:
                                # Generic model: try to get penultimate layer
                                outputs = model(inputs, output_hidden_states=True)
                                if hasattr(outputs, 'hidden_states') and outputs.hidden_states:
                                    embeddings = outputs.hidden_states[-1][:, 0, :].cpu().numpy()
                                else:
                                    continue
                        else:
                            # Direct module call
                            outputs = pl_module(inputs, output_hidden_states=True)
                            if hasattr(outputs, 'hidden_states') and outputs.hidden_states:
                                embeddings = outputs.hidden_states[-1][:, 0, :].cpu().numpy()
                            else:
                                continue
                        
                        embeddings_list.append(embeddings)
                        samples_collected += len(embeddings)
                        
                        # Clear GPU memory periodically
                        if batch_idx % 3 == 0:
                            del outputs
                            torch.cuda.empty_cache()
                            
                    except Exception as e:
                        print(f"[VizCallback] Embedding extraction failed for batch {batch_idx}: {e}")
                        continue
            
            if embeddings_list:
                all_embeddings = np.concatenate(embeddings_list, axis=0)[:self.num_viz_samples]
                print(f"[VizCallback] Extracted {len(all_embeddings)} embeddings with shape {all_embeddings.shape}")
                return all_embeddings
                
        except Exception as e:
            print(f"[VizCallback] Could not extract embeddings: {e}")
        
        return None
    
    def _log_to_wandb(self, logger, epoch: int):
        """Log visualizations to WandB."""
        if logger is None or not hasattr(logger, 'experiment'):
            return
        
        try:
            import wandb
            
            log_dict = {}
            
            # Map frame collections to WandB keys
            frame_mappings = [
                ('distribution_frames', 'viz/distribution'),
                ('activation_frames', 'viz/activation'),
                ('attention_frames', 'viz/attention'),
                ('attention_heatmap_frames', 'viz/attention_heatmap'),
                ('embedding_frames', 'viz/embedding'),
                ('collinearity_frames', 'viz/collinearity'),
            ]
            
            for frame_attr, wandb_key in frame_mappings:
                frames = getattr(self, frame_attr, None)
                if frames and len(frames) > 0:
                    log_dict[wandb_key] = wandb.Image(frames[-1])
            
            if log_dict:
                logger.experiment.log(log_dict)
        except Exception as e:
            pass
    
    def on_train_end(self, trainer, pl_module):
        """Create final GIFs at training end."""
        gif_paths = []
        
        # Use temporary directory if not saving locally
        import tempfile
        temp_dir = None if self.save_local else tempfile.mkdtemp(prefix="viz_")
        save_dir = self.save_dir if self.save_local else Path(temp_dir)
        
        # Define all visualizations to process
        viz_configs = [
            ('distribution_frames', 'prediction_distribution_evolution.gif', 'Distribution', 
             'viz/distribution_gif', None),
            ('activation_frames', 'activation_statistics_evolution.gif', 'Activation', 
             'viz/activation_gif', None),
            ('attention_frames', 'attention_pattern_evolution.gif', 'Attention', 
             'viz/attention_gif', None),
            ('attention_heatmap_frames', 'attention_heatmap_evolution.gif', 'Attention Heatmap', 
             'viz/attention_heatmap_gif', None),
            ('embedding_frames', 'embedding_space_evolution.gif', 'Embedding', 
             'viz/embedding_gif', 'final_embedding_space.png'),
            ('collinearity_frames', 'embedding_collinearity_evolution.gif', 'Collinearity', 
             'viz/collinearity_gif', 'final_embedding_collinearity.png'),
        ]
        
        for frame_attr, gif_filename, viz_name, wandb_key, final_png in viz_configs:
            frames = getattr(self, frame_attr, None)
            if frames and len(frames) > 0:
                # Save GIF
                gif_path = save_dir / gif_filename
                save_gif(list(frames), gif_path, self.gif_duration, viz_name)
                gif_paths.append((wandb_key, gif_path))
                
                # Save final frame as PNG if specified
                if final_png:
                    png_path = save_dir / final_png
                    frames[-1].save(png_path)
        
        # Upload GIFs to WandB
        self._upload_gifs_to_wandb(trainer.logger, gif_paths)
        
        # Cleanup temporary directory if used
        if temp_dir:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    def _upload_gifs_to_wandb(self, logger, gif_paths: list):
        """Upload GIF files to WandB."""
        if logger is None or not hasattr(logger, 'experiment'):
            return
        
        if not gif_paths:
            return
        
        try:
            import wandb
            
            log_dict = {}
            for wandb_key, gif_path in gif_paths:
                if gif_path.exists():
                    log_dict[wandb_key] = wandb.Video(str(gif_path), fps=4, format="gif")
            
            if log_dict:
                logger.experiment.log(log_dict)
        except Exception as e:
            pass
