"""Training callbacks for model warmup and freezing strategies."""

import lightning as L


class PreprocessorFreezeCallback(L.Callback):
    """Callback to freeze/unfreeze preprocessor parameters during training.
    
    The preprocessor (e.g., ZCA whitening layer) can be frozen for warmup,
    then unfrozen for fine-tuning, or kept permanently frozen.
    
    Args:
        freeze_epochs: Controls freezing behavior:
                      - freeze_epochs > 0: Freeze for N epochs, then unfreeze
                      - freeze_epochs = -1: Permanently frozen (never unfreeze)
                      - freeze_epochs = 0: Never frozen (always trainable)
    
    Example:
        >>> # Freeze for 5 epochs then unfreeze
        >>> callback = PreprocessorFreezeCallback(freeze_epochs=5)
        >>> 
        >>> # Permanently frozen (feature extractor)
        >>> callback = PreprocessorFreezeCallback(freeze_epochs=-1)
    """
    
    def __init__(self, freeze_epochs: int = 0):
        super().__init__()
        self.freeze_epochs = freeze_epochs
        self._unfrozen = False
    
    def on_train_epoch_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Check if we should unfreeze preprocessor at the start of each epoch."""
        current_epoch = trainer.current_epoch
        
        # If freeze_epochs == -1, keep frozen permanently (never unfreeze)
        if self.freeze_epochs == -1:
            return
        
        # Only unfreeze once when reaching freeze_epochs (and freeze_epochs > 0)
        if self.freeze_epochs > 0 and current_epoch >= self.freeze_epochs and not self._unfrozen:
            if hasattr(pl_module.model, 'set_preprocessor_trainable'):
                pl_module.model.set_preprocessor_trainable(True)
                self._unfrozen = True
                print(f"\n[PreprocessorFreezeCallback] Epoch {current_epoch}: "
                      f"Unfreezing preprocessor for fine-tuning")
            else:
                print(f"\n[PreprocessorFreezeCallback] Warning: Model does not have "
                      f"'set_preprocessor_trainable' method")
    
    def on_train_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Ensure preprocessor is frozen at the start of training."""
        if self.freeze_epochs != 0:  # Freeze if not 0 (either temporary or permanent)
            if hasattr(pl_module.model, 'set_preprocessor_trainable'):
                pl_module.model.set_preprocessor_trainable(False)
                if self.freeze_epochs == -1:
                    print(f"[PreprocessorFreezeCallback] Preprocessor PERMANENTLY FROZEN")
                else:
                    print(f"[PreprocessorFreezeCallback] Freezing preprocessor for "
                          f"first {self.freeze_epochs} epochs")
            else:
                print(f"[PreprocessorFreezeCallback] Warning: Model does not have "
                      f"'set_preprocessor_trainable' method")


__all__ = ['PreprocessorFreezeCallback']
