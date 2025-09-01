# =============================
# callbacks_pca_warm.py
# =============================
from __future__ import annotations
from typing import Optional, Dict, Any

import torch
import lightning as L
from lightning.pytorch import Trainer, LightningModule, Callback

from src.pca_warm import fit_pca, init_qkv_from_pca
from src.cka import compute_cka


# L.pytorch.callbacks.ModelCheckpoint(dirpath= SAVE_PATH, filename='{epoch}-{acc_valid:.0f}', save_top_k=1, monitor=f'val_{monitor_name}', mode=monitor_mode)


class PCAWarmStartCallback(Callback):
    def __init__(
        self,
        attn_module_path: str,  # e.g., 'model.encoder.layers.0.attn'
        r: int = 32,
        robust: bool = False,
        kernel: Optional[str] = None,
        nystrom_m: Optional[int] = None,
        whiten: bool = False,
        trigger_epoch: int = 0,
    ):
        super().__init__()
        self.path = attn_module_path
        self.r = r
        self.robust = robust
        self.kernel = kernel
        self.nystrom_m = nystrom_m
        self.whiten = whiten
        self.trigger_epoch = trigger_epoch
        self._done = False

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if self.trigger_epoch > 0:
            return
        self._apply(pl_module)

    def on_train_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if (not self._done) and (trainer.current_epoch >= self.trigger_epoch):
            self._apply(pl_module)

    def _apply(self, pl_module: LightningModule):
        try:
            attn = _resolve_attr(pl_module, self.path)
        except Exception as e:
            pl_module.print(f"[PCAWarm] Cannot resolve module '{self.path}': {e}")
            return
        # gather a small cache of training samples from dataloader to fit PCA
        X_batch = _peek_batch(pl_module)
        if X_batch is None:
            pl_module.print("[PCAWarm] Warning: could not fetch a batch to fit PCA.")
            return
        X = X_batch.detach().flatten(0, 1)  # [n, d] assume last dim is features
        res = fit_pca(X, r=self.r, robust=self.robust, kernel=self.kernel, nystrom_m=self.nystrom_m)
        assigned = init_qkv_from_pca(attn, res['U'], mu=res['mu'], whiten=self.whiten, eigvals=res.get('eigvals', None))
        self._done = True
        pl_module.print(f"[PCAWarm] Initialized {list(assigned.keys())} with r={self.r} (kernel={self.kernel})")


def _resolve_attr(root: torch.nn.Module, path: str) -> torch.nn.Module:
    mod = root
    for p in path.split('.'):
        mod = getattr(mod, p)
    return mod


def _peek_batch(pl_module: LightningModule):
    dl = pl_module.trainer.datamodule.train_dataloader() if pl_module.trainer.datamodule else None
    if dl is None:
        return None
    try:
        batch = next(iter(dl))
    except Exception:
        return None
    # try to standardize: (x, y) or dict
    if isinstance(batch, (list, tuple)):
        x = batch[0]
    elif isinstance(batch, dict):
        x = batch.get('x', None)
    else:
        x = batch
    return x if isinstance(x, torch.Tensor) else None


class CKAProbeCallback(Callback):
    def __init__(self, attn_module_path: str, r: int = 32, every_n_epochs: int = 1, kernel: str = 'linear'):
        super().__init__()
        self.path = attn_module_path
        self.r = r
        self.every = every_n_epochs
        self.kernel = kernel

    @torch.no_grad()
    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if trainer.current_epoch % self.every != 0:
            return
        try:
            attn = _resolve_attr(pl_module, self.path)
        except Exception as e:
            pl_module.print(f"[CKAProbe] resolve error: {e}")
            return
        # compute PCA basis on a small val batch
        X_batch = _peek_val_batch(pl_module)
        if X_batch is None:
            return
        X = X_batch.detach().flatten(0, 1)
        pca = fit_pca(X, r=self.r)
        # collect current Q/K features on the same batch (forward hook)
        Q, K = _extract_qk(attn, X_batch)
        # project tokens to d_model features (flatten batch, tokens)
        n = min(Q.shape[0], K.shape[0])
        A = Q[:n]
        B = X[:n] @ pca['U']  # PCA scores as target reps
        score = compute_cka(A, B, kernel=self.kernel, debiased=True)
        pl_module.log("cka_q_vs_pca", score, prog_bar=True)
        pl_module.print(f"[CKAProbe] epoch={trainer.current_epoch} CKA(Q, PCA)={score:.4f}")


def _peek_val_batch(pl_module: LightningModule):
    dl = pl_module.trainer.datamodule.val_dataloader() if pl_module.trainer.datamodule else None
    if dl is None:
        return None
    try:
        batch = next(iter(dl))
    except Exception:
        return None
    if isinstance(batch, (list, tuple)):
        x = batch[0]
    elif isinstance(batch, dict):
        x = batch.get('x', None)
    else:
        x = batch
    return x if isinstance(x, torch.Tensor) else None


def _extract_qk(attn_module: torch.nn.Module, x_tokens: torch.Tensor):
    """Lightweight hook to fetch Q/K before softmax for the provided attention.
    This is highly model-specific; here we assume the module exposes a method get_qk or attributes.
    Replace with your model's actual API if available.
    """
    # Fallback: return token embeddings as both reps (so CKA still runs)
    if hasattr(attn_module, 'get_qk'):
        Q, K = attn_module.get_qk(x_tokens)
        Q = Q.flatten(0, 1)
        K = K.flatten(0, 1)
        return Q, K
    return x_tokens.flatten(0, 1), x_tokens.flatten(0, 1)



# import lightning as pl
# from pca_warm0 import Tokenizer1DConfig, WarmStartConfig, pca_warm_start_model

# class PCAWarmStartCallback(pl.Callback):
#     """Run PCA warm-start at the beginning of training using the train dataloader."""
#     def __init__(self, tokenizer_cfg: Tokenizer1DConfig, warm_cfg: WarmStartConfig, 
#                  dataloader_max_batches: int = 50, center: bool = True, whiten: bool = False):
#         super().__init__()
#         self.tokenizer_cfg = tokenizer_cfg
#         self.warm_cfg = warm_cfg
#         self.max_batches = dataloader_max_batches
#         self.center = center
#         self.whiten = whiten

#     def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
#         # Resolve a training dataloader
#         if trainer.datamodule is not None and hasattr(trainer.datamodule, "train_dataloader"):
#             train_loader = trainer.datamodule.train_dataloader()
#         elif hasattr(trainer, "train_dataloader") and trainer.train_dataloader is not None:
#             train_loader = trainer.train_dataloader
#         else:
#             raise RuntimeError("PCAWarmStartCallback: cannot find train dataloader.")
#         # Limit batches for quick PCA
#         def limited_loader():
#             count = 0
#             for b in train_loader:
#                 yield b
#                 count += 1
#                 if self.max_batches and count >= self.max_batches:
#                     break
#         model = getattr(pl_module, 'model', pl_module)
#         sub = pca_warm_start_model(model, limited_loader(), self.tokenizer_cfg, self.warm_cfg,
#                                    max_batches=self.max_batches, center=self.center, whiten=self.whiten)
#         # Optional: log eigen spectrum if logger supports .log_metrics
#         try:
#             if trainer.logger is not None and getattr(sub, 'explained_variance', None) is not None:
#                 ev = sub.explained_variance
#                 trainer.logger.log_metrics({f"pca/ev_{i}": float(v) for i, v in enumerate(ev)}, step=0)
#         except Exception:
#             pass
