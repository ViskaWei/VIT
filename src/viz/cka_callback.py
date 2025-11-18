"""CKA (Centered Kernel Alignment) analysis callback for layer similarity monitoring."""

import numpy as np
import torch
import lightning as L
from pathlib import Path
from typing import Optional, List

from .cka_utils import (
    extract_layer_representations,
    compute_diagonal_cka
)


class CKACallback(L.Callback):
    """
    Callback for CKA (Centered Kernel Alignment) analysis.
    Monitors how much each layer changes during training.
    """
    
    def __init__(
        self,
        save_dir: str = './results/viz',
        log_every_n_epochs: int = 5,
        log_every_n_steps: Optional[int] = None,  # NEW: step-based logging
        cka_layers: Optional[List[str]] = None,
    ):
        super().__init__()
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_every_n_epochs = log_every_n_epochs
        self.log_every_n_steps = log_every_n_steps  # If set, overrides epoch-based
        self.cka_layers = cka_layers
        self.initial_representations = None
        self.cka_scores_history = []
    
    def _should_log(self, trainer):
        """Check if we should log at this epoch/step."""
        # Step-based logging takes priority
        if self.log_every_n_steps is not None:
            return trainer.global_step % self.log_every_n_steps == 0
        # Fall back to epoch-based
        return trainer.current_epoch % self.log_every_n_epochs == 0
    
    def on_validation_epoch_start(self, trainer, pl_module):
        """Save initial representations on first logging epoch."""
        if trainer.sanity_checking:
            return
        
        # Save initial representations on first logging epoch for CKA analysis
        # This happens when: (1) not saved yet AND (2) it's a logging epoch
        if self.initial_representations is None and self._should_log(trainer):
            self._save_initial_representations(pl_module, trainer.val_dataloaders)
    
    def on_validation_epoch_end(self, trainer, pl_module):
        """Compute CKA scores and generate report."""
        if trainer.sanity_checking:
            return
        
        # Check if we should log
        if not self._should_log(trainer):
            return
        
        current_epoch = trainer.current_epoch
        
        # Compute CKA analysis
        if self.initial_representations is not None:
            self._compute_and_log_cka(pl_module, trainer, current_epoch)
            
            # Generate and upload CKA report after computing scores
            if self.cka_scores_history:
                self._generate_cka_report(trainer.logger)
    
    def on_train_end(self, trainer, pl_module):
        """Generate CKA summary report."""
        if self.cka_scores_history:
            self._generate_cka_report(trainer.logger)
    
    def _save_initial_representations(self, pl_module, dataloader):
        """Save initial model representations for CKA analysis."""
        print("[CKA] Saving initial representations...")
        
        model = pl_module.model if hasattr(pl_module, 'model') else pl_module
        
        # Determine layers to analyze
        if self.cka_layers is None:
            # Auto-select important layers from the actual model
            # Only select layers that will actually produce outputs (leaf modules)
            all_layer_names = []
            for name, module in model.named_modules():
                # Check if it's a leaf module (has no children)
                if name and len(list(module.children())) == 0:
                    all_layer_names.append(name)
            
            self.cka_layers = []
            for name in all_layer_names:
                # Select key computational layers: query/key/value, dense layers in attention/mlp
                if any(key in name for key in ['query', 'key', 'value', '.dense', 'pooler', 'head']):
                    if not any(skip in name for skip in ['norm', 'dropout', 'bias', 'LayerNorm', 'embedding']):
                        self.cka_layers.append(name)
            
            # Limit to reasonable number (prefer deeper layers)
            self.cka_layers = self.cka_layers[:12]
        
        if not self.cka_layers:
            print("[CKA] No layers selected for CKA analysis")
            return
        
        print(f"[CKA] Monitoring {len(self.cka_layers)} layers")
        
        self.initial_representations = extract_layer_representations(
            model, dataloader, self.cka_layers, 
            device=pl_module.device, max_samples=500
        )
        
        print(f"[CKA] Saved representations for {len(self.initial_representations)} layers")
    
    def _compute_and_log_cka(self, pl_module, trainer, epoch: int):
        """Compute CKA between initial and current representations."""
        # For epoch 0, initial == current, so CKA should be 1.0
        # Reuse initial representations instead of re-extracting to avoid randomness
        if epoch == 0:
            current_representations = self.initial_representations
        else:
            # Extract current representations
            model = pl_module.model if hasattr(pl_module, 'model') else pl_module
            current_representations = extract_layer_representations(
                model, trainer.val_dataloaders, self.cka_layers,
                device=pl_module.device, max_samples=500
            )
        
        # Compute diagonal CKA (same layer, initial vs trained)
        cka_scores = compute_diagonal_cka(
            self.initial_representations, 
            current_representations,
            use_rbf=False
        )
        
        # Store scores
        self.cka_scores_history.append({
            'epoch': epoch,
            'scores': cka_scores
        })
        
        # Log to WandB only (no console output)
        mean_cka = np.mean(list(cka_scores.values()))
        high_cka_count = sum(1 for score in cka_scores.values() if score >= 0.95)
        
        if trainer.logger and hasattr(trainer.logger, 'experiment'):
            try:
                import wandb
                log_dict = {f"cka/{name}": score for name, score in cka_scores.items()}
                log_dict['cka/mean'] = mean_cka
                log_dict['cka/unchanged_count'] = high_cka_count
                wandb.log(log_dict)
            except:
                pass
    
    def _generate_cka_report(self, logger=None):
        """Generate final CKA analysis report."""
        if not self.cka_scores_history:
            return
            
        import matplotlib.pyplot as plt
        
        print("\n[CKA] Generating final analysis report...")
        
        # Extract data
        epochs = [entry['epoch'] for entry in self.cka_scores_history]
        all_layers = sorted(self.cka_scores_history[0]['scores'].keys())
        
        # Create figure with 2 subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot 1: CKA evolution over epochs for each layer
        for layer in all_layers[:10]:  # Limit to top 10 layers for clarity
            scores = [entry['scores'].get(layer, 0) for entry in self.cka_scores_history]
            ax1.plot(epochs, scores, 'o-', label=layer.split('.')[-1], alpha=0.7, lw=2)
        
        ax1.axhline(0.95, color='red', linestyle='--', lw=2, label='Threshold (0.95)')
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('CKA Score (Initial vs Current)', fontsize=12)
        ax1.set_title('Layer Similarity Evolution\n(High CKA = Little Learning)', fontsize=14)
        ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([0, 1.05])
        
        # Plot 2: Final CKA scores (bar chart)
        final_scores = self.cka_scores_history[-1]['scores']
        sorted_layers = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        layer_names = [name.split('.')[-1] for name, _ in sorted_layers[:15]]
        scores = [score for _, score in sorted_layers[:15]]
        
        colors = ['red' if s >= 0.95 else 'orange' if s >= 0.80 else 'green' for s in scores]
        ax2.barh(range(len(layer_names)), scores, color=colors, alpha=0.7, edgecolor='black')
        ax2.set_yticks(range(len(layer_names)))
        ax2.set_yticklabels(layer_names, fontsize=10)
        ax2.set_xlabel('CKA Score', fontsize=12)
        ax2.set_title(f'Final Layer Similarity (Epoch {epochs[-1]})', fontsize=14)
        ax2.axvline(0.95, color='red', linestyle='--', lw=2, label='Warning (≥0.95)')
        ax2.axvline(0.80, color='orange', linestyle=':', lw=1.5, label='Low Change (≥0.80)')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3, axis='x')
        ax2.set_xlim([0, 1.05])
        
        plt.tight_layout()
        
        # Check if wandb is enabled
        has_wandb = logger and hasattr(logger, 'experiment')
        
        # If wandb is enabled, only log to wandb (don't save locally)
        if has_wandb:
            try:
                import wandb
                wandb.log({'cka/analysis_report': wandb.Image(fig)})
                print(f"  ✓ Logged CKA report to WandB")
            except Exception as e:
                print(f"  ⚠ Could not log CKA report to WandB: {e}")
        else:
            # Only save locally if wandb is not enabled
            report_path = self.save_dir / 'cka_analysis_report.png'
            fig.savefig(report_path, dpi=150, bbox_inches='tight')
            print(f"  ✓ Saved CKA report: {report_path}")
        
        plt.close(fig)
