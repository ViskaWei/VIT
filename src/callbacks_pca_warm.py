
import pytorch_lightning as pl
from pca_warm import Tokenizer1DConfig, WarmStartConfig, pca_warm_start_model

class PCAWarmStartCallback(pl.Callback):
    """Run PCA warm-start at the beginning of training using the train dataloader."""
    def __init__(self, tokenizer_cfg: Tokenizer1DConfig, warm_cfg: WarmStartConfig, 
                 dataloader_max_batches: int = 50, center: bool = True, whiten: bool = False):
        super().__init__()
        self.tokenizer_cfg = tokenizer_cfg
        self.warm_cfg = warm_cfg
        self.max_batches = dataloader_max_batches
        self.center = center
        self.whiten = whiten

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        # Resolve a training dataloader
        if trainer.datamodule is not None and hasattr(trainer.datamodule, "train_dataloader"):
            train_loader = trainer.datamodule.train_dataloader()
        elif hasattr(trainer, "train_dataloader") and trainer.train_dataloader is not None:
            train_loader = trainer.train_dataloader
        else:
            raise RuntimeError("PCAWarmStartCallback: cannot find train dataloader.")
        # Limit batches for quick PCA
        def limited_loader():
            count = 0
            for b in train_loader:
                yield b
                count += 1
                if self.max_batches and count >= self.max_batches:
                    break
        model = getattr(pl_module, 'model', pl_module)
        sub = pca_warm_start_model(model, limited_loader(), self.tokenizer_cfg, self.warm_cfg,
                                   max_batches=self.max_batches, center=self.center, whiten=self.whiten)
        # Optional: log eigen spectrum if logger supports .log_metrics
        try:
            if trainer.logger is not None and getattr(sub, 'explained_variance', None) is not None:
                ev = sub.explained_variance
                trainer.logger.log_metrics({f"pca/ev_{i}": float(v) for i, v in enumerate(ev)}, step=0)
        except Exception:
            pass
