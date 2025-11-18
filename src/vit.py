import os
import torch
from torchmetrics import Accuracy, MeanAbsoluteError, MeanSquaredError, R2Score
import lightning as L

from src.basemodule import BaseLightningModule, BaseTrainer, BaseDataModule
from src.dataloader import ClassSpecDataset, RegSpecDataset
from src.prepca.callbacks import PreprocessorFreezeCallback
# from src.callbacks_pca_warm import PCAWarmStartCallback, CKAProbeCallback

# Use env override for local W&B run files
# Falls back to ./wandb in the project if not set
SAVE_DIR = os.environ.get('WANDB_DIR', './wandb')
# Checkpoint directory: read from env CHECKPOINT_DIR, fallback to ./checkpoints
CKPT_DIR = os.environ.get('CKPT_DIR', './checkpoints')
# Plot save directory
PLOT_DIR = os.environ.get('PLOT_DIR', './results/test_plots')
# MASK_PATH = './bosz50000_mask.npy'

#region --DATA-----------------------------------------------------------
def _normalize_task(config):
    """Return ('cls'|'reg') from config supporting legacy keys."""
    m = (config.get('model', {}) or {})
    task = (m.get('task_type') or m.get('task') or 'cls').lower()
    if task in ('classification', 'cls', 'class'):
        return 'cls'
    return 'reg'
#endregion --DATA-----------------------------------------------------------
#region --DATAMODULE-----------------------------------------------------------
class ViTDataModule(BaseDataModule):
    @classmethod
    def from_config(cls, config, test_data=False):
        task = _normalize_task(config)
        if test_data:
            raise ValueError("test_data=True is not supported. TestDataset has been removed.")
        if task == 'reg':
            dataset_cls = RegSpecDataset
            print('Using RegSpec Dataset (regression)')
        else:
            dataset_cls = ClassSpecDataset
            print('Using ClassSpec Dataset (classification)')
        return super().from_config(dataset_cls=dataset_cls, config=config)

    def setup_test_dataset(self, stage):
        d = self.dataset_cls.from_config(self.config)
        # If training dataset computed label normalization stats, propagate them
        if hasattr(self, 'train') and hasattr(self.train, 'label_norm') and getattr(self.train, 'label_norm', 'none') != 'none':
            for k in ('label_norm', 'label_mean', 'label_std', 'label_min', 'label_max'):
                if hasattr(self.train, k):
                    setattr(d, k, getattr(self.train, k))
        return d
#endregion --DATAMODULE-----------------------------------------------------------

#region MODEL-----------------------------------------------------------
from src.models import get_model

#region --TRAINER-----------------------------------------------------------
class ViTLModule(BaseLightningModule):
    def __init__(self, model=None, config = {}):
        model = model or self.get_model(config)
        super().__init__(model=model, config=config)
        self.save_hyperparameters(ignore=['model'])
        self.task_type = _normalize_task(config)
        # Get noise_level from config, similar to blindspot.py
        self.noise_level = config.get('noise', {}).get('noise_level', 0.0)
        if self.task_type == 'cls':
            self.accuracy = Accuracy(task='multiclass', num_classes=config['model']['num_labels'])
            self.monitor_metric = 'acc'
        elif self.task_type == 'reg':
            self.mae = MeanAbsoluteError()
            self.mse = MeanSquaredError()
            self.r2 = R2Score()
            self.monitor_metric = 'mae'

    def get_model(self, config):
        return get_model(config)

    def forward(self, flux, labels, loss_only=True):
        """Forward wrapper returning loss or full outputs from HF model."""
        outputs = self.model(flux, labels=labels)
        return outputs.loss if loss_only else outputs
        
    def training_step(self, batch, batch_idx):
        # Training: batch is (flux, error, labels) - generate noise on the fly
        flux, error, labels = batch
        if self.noise_level > 0:
            noisy = flux + torch.randn_like(flux) * error * self.noise_level
            loss = self(noisy, labels, loss_only=True)
        else:
            loss = self(flux, labels, loss_only=True)
        self.log(f'{self.loss_name}_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    def _shared_eval_step(self, batch, prefix):
        # Val/Test: batch can be (noisy, flux, error, labels) with pre-generated noise
        # or (flux, error, labels) without noise
        if len(batch) == 4:
            # Pre-generated noisy data from dataset (like blindspot.py)
            noisy, flux, error, labels = batch
            if self.noise_level > 0:
                # Use pre-generated noisy data
                outputs = self.forward(noisy, labels, loss_only=False)
            else:
                # noise_level is 0, use clean flux
                outputs = self.forward(flux, labels, loss_only=False)
        else:
            # No pre-generated noise (fallback or training dataset reused)
            flux, error, labels = batch
            outputs = self.forward(flux, labels, loss_only=False)
        
        loss = outputs.loss
        self.log(f'{prefix}_{self.loss_name}_loss', loss, on_step=False, on_epoch=True)
        if self.task_type == 'cls':
            acc = self.accuracy(outputs.logits, labels)
            self.log(f'{prefix}_acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        elif self.task_type == 'reg':
            preds = outputs.logits.squeeze()
            mae = self.mae(preds, labels)
            mse = self.mse(preds, labels)
            r2 = self.r2(preds, labels)
            self.log(f'{prefix}_mae', mae, on_step=False, on_epoch=True)
            self.log(f'{prefix}_mse', mse, on_step=False, on_epoch=True)
            self.log(f'{prefix}_r2', r2, on_step=False, on_epoch=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._shared_eval_step(batch, 'val')
        
        # Collect predictions for epoch-level metrics (regression only)
        if self.task_type == 'reg' and hasattr(self, 'val_dict'):
            # Extract labels from batch
            labels = batch[-1]  # Last element is always labels
            # Get predictions (already computed in _shared_eval_step, cached in metrics)
            # Use mae.update's internal state to avoid re-computing
            preds = self.mae.compute()  # This won't work, need to pass through
            # Actually, let's just recompute once here to get preds
            if len(batch) == 4:
                noisy, flux, error, labels = batch
                inputs = noisy if self.noise_level > 0 else flux
            else:
                flux, error, labels = batch
                inputs = flux
            with torch.no_grad():
                outputs = self.forward(inputs, labels, loss_only=False)
                preds = outputs.logits.squeeze()
            self.val_dict['preds'].append(preds.detach().cpu())
            self.val_dict['labels'].append(labels.detach().cpu())
        
        return loss
    
    def on_validation_start(self):
        """Initialize collection for epoch-level metrics"""
        if self.task_type == 'reg':
            self.val_dict = {'preds': [], 'labels': []}
    
    def on_validation_epoch_end(self):
        """Calculate epoch-level regression metrics"""
        if self.task_type != 'reg' or not hasattr(self, 'val_dict') or not self.val_dict['preds']:
            return
        
        import numpy as np
        all_preds = torch.cat(self.val_dict['preds'], dim=0).numpy()
        all_labels = torch.cat(self.val_dict['labels'], dim=0).numpy()
        
        # Ensure 2D shape
        if all_preds.ndim == 1:
            all_preds = all_preds.reshape(-1, 1)
            all_labels = all_labels.reshape(-1, 1)
        
        # Calculate for each output
        for i in range(all_preds.shape[1]):
            residuals = all_preds[:, i] - all_labels[:, i]
            bias_median = float(np.median(residuals))
            p90 = float(np.percentile(np.abs(residuals), 90))
            
            # Linear fit: pred = a + b * label
            coeffs = np.polyfit(all_labels[:, i], all_preds[:, i], 1)
            beta, a = float(coeffs[0]), float(coeffs[1])
            
            # Log (single output uses simple names)
            suffix = '' if all_preds.shape[1] == 1 else f'_{i}'
            self.log(f'val_bias_median{suffix}', bias_median, on_epoch=True)
            self.log(f'val_p90{suffix}', p90, on_epoch=True)
            self.log(f'val_beta{suffix}', beta, on_epoch=True)
        
        self.val_dict = {'preds': [], 'labels': []}

    def on_test_start(self):
        """Initialize dict for collecting test predictions and labels (regression only)"""
        if self.task_type == 'reg':
            self.test_dict = {'preds': [], 'labels': []}

    def test_step(self, batch, batch_idx):
        loss = self._shared_eval_step(batch, 'test')
        
        # Collect predictions and labels for regression plotting
        if self.task_type == 'reg':
            # Extract data from batch
            if len(batch) == 4:
                noisy, flux, error, labels = batch
                inputs = noisy if self.noise_level > 0 else flux
            else:
                flux, error, labels = batch
                inputs = flux
            
            # Get predictions
            outputs = self.forward(inputs, labels, loss_only=False)
            preds = outputs.logits.squeeze()
            
            # Store for plotting
            self.test_dict['preds'].append(preds.detach().cpu())
            self.test_dict['labels'].append(labels.detach().cpu())
        
        return loss

    def on_test_epoch_end(self):
        """Generate evaluation plots after testing (regression only)"""
        if self.task_type != 'reg' or not self.test_dict['preds']:
            return
        
        # Concatenate all predictions and labels
        import numpy as np
        all_preds = torch.cat(self.test_dict['preds'], dim=0).numpy()
        all_labels = torch.cat(self.test_dict['labels'], dim=0).numpy()
        
        # Get parameter names from config
        n_outputs = all_preds.shape[1] if len(all_preds.shape) > 1 else 1
        param_names = self.config.get('model', {}).get('param_names', ['Teff', 'log_g', 'M_H'][:n_outputs])
        
        # Get normalization parameters from datamodule's test dataset
        label_norm = None
        label_mean = None
        label_std = None
        label_min = None
        label_max = None
        
        try:
            # Access test dataset through trainer.datamodule
            if hasattr(self.trainer, 'datamodule') and hasattr(self.trainer.datamodule, 'test'):
                test_dataset = self.trainer.datamodule.test
                label_norm = getattr(test_dataset, 'label_norm', None)
                if label_norm in ('standard', 'zscore'):
                    label_mean = getattr(test_dataset, 'label_mean', None)
                    label_std = getattr(test_dataset, 'label_std', None)
                elif label_norm == 'minmax':
                    label_min = getattr(test_dataset, 'label_min', None)
                    label_max = getattr(test_dataset, 'label_max', None)
                print(f"[on_test_epoch_end] Retrieved normalization params: norm={label_norm}")
        except Exception as e:
            print(f"[on_test_epoch_end] Could not retrieve normalization params: {e}")
        
        # Determine if we should save locally
        # If wandb is enabled, don't save locally. Otherwise, always save locally.
        has_logger = self.logger is not None and hasattr(self.logger, 'experiment')
        save_enabled = self.config.get('train', {}).get('save', False)
        
        # If wandb is enabled, use temp directory (will be cleaned up)
        # If wandb is not enabled, always save locally regardless of save_enabled flag
        import tempfile
        if has_logger:
            # Wandb enabled: use temp directory, plots will only go to wandb
            plot_dir = tempfile.mkdtemp(prefix="test_plots_")
            print(f"[on_test_epoch_end] Using temp directory for plots (will upload to WandB only): {plot_dir}")
            should_save_local = False
        else:
            # Wandb not enabled: always save locally
            plot_dir = PLOT_DIR
            should_save_local = True
            print(f"[on_test_epoch_end] Saving plots locally to: {plot_dir}")
        
        # Use RegressionPlotter for all visualizations
        from src.viz import RegressionPlotter
        plotter = RegressionPlotter(
            predictions=all_preds,
            labels=all_labels,
            param_names=param_names,
            logger=self.logger,
            save_dir=plot_dir,
            label_norm=label_norm,
            label_mean=label_mean,
            label_std=label_std,
            label_min=label_min,
            label_max=label_max
        )
        
        # Generate all plots (use quick_mode=True for faster execution)
        quick_mode = self.config.get('plotting', {}).get('quick_mode', False)
        plotter.generate_all_plots(quick_mode=quick_mode)
        
        # Cleanup temporary directory if used (only when wandb is enabled)
        if not should_save_local:
            import shutil
            shutil.rmtree(plot_dir, ignore_errors=True)
            print(f"[on_test_epoch_end] Cleaned up temporary test plots")

    def on_train_start(self):
        # Log explained variance (if GlobalAttnViT with PCA stats and r provided)
        try:
            attn = getattr(self.model, 'attn', None)
            if attn is not None and hasattr(attn, 'explained_variance_at_r') and attn.explained_variance_at_r is not None:
                self.log('pca_explained_variance_at_r', attn.explained_variance_at_r, on_step=False, on_epoch=True, prog_bar=True)
        except Exception:
            pass

        # Apply initial Q/K freeze state at epoch 0 if configured
        try:
            if hasattr(self.model, 'apply_qk_freeze'):
                frozen = bool(self.model.apply_qk_freeze(current_epoch=0))
                self.log('qk_frozen', int(frozen), on_step=False, on_epoch=True)
        except Exception:
            pass

        # Apply initial embed freeze state at epoch 0 if configured
        try:
            if hasattr(self.model, 'apply_embed_freeze'):
                frozen = bool(self.model.apply_embed_freeze(current_epoch=0))
                self.log('embed_frozen', int(frozen), on_step=False, on_epoch=True)
        except Exception:
            pass

    def on_train_epoch_start(self):
        # Keep BaseLightningModule behavior (LR logging)
        super().on_train_epoch_start()
        # Enforce Q/K freeze schedule if available on the model
        try:
            if hasattr(self.model, 'apply_qk_freeze'):
                frozen = bool(self.model.apply_qk_freeze(current_epoch=self.current_epoch))
                # Log an indicator for monitoring
                self.log('qk_frozen', int(frozen), on_step=False, on_epoch=True, prog_bar=False)
        except Exception:
            pass
        # Enforce embed freeze schedule if available on the model
        try:
            if hasattr(self.model, 'apply_embed_freeze'):
                frozen = bool(self.model.apply_embed_freeze(current_epoch=self.current_epoch))
                self.log('embed_frozen', int(frozen), on_step=False, on_epoch=True, prog_bar=False)
        except Exception:
            pass
    
    # def test_step(self, batch, batch_idx):
    #     # Compute metrics as usual
    #     flux, _, labels = batch
    #     outputs = self.forward(flux, labels, loss_only=False)
    #     loss = outputs.loss
    #     self.log(f'test_{self.loss_name}_loss', loss, on_step=False, on_epoch=True)
    #     if self.task_type == 'cls':
    #         acc = self.accuracy(outputs.logits, labels)
    #         self.log('test_acc', acc, on_step=False, on_epoch=True, prog_bar=True)
    #         return {"loss": loss}
    #     elif self.task_type == 'reg':
    #         preds = outputs.logits.squeeze()
    #         mae = self.mae(preds, labels)
    #         r2 = self.r2(preds, labels)
    #         self.log('test_mae', mae, on_step=False, on_epoch=True)
    #         self.log('test_r2', r2, on_step=False, on_epoch=True)
    #         # Return preds and labels for callbacks to post-process/plot
    #         return {"loss": loss, "preds": preds.detach().cpu(), "labels": labels.detach().cpu()}
    #     return {"loss": loss}
#endregion --TRAINER-----------------------------------------------------------

class SpecTrainer():
    def __init__(self, config, logger, num_gpus=None, sweep=False, monitor_name='acc', monitor_mode='max') -> None:
        if sweep: num_gpus = 1
        patience = 100 if sweep else 500
        task_type = _normalize_task(config)
        if task_type == 'cls':
            monitor_name, monitor_mode = 'acc', 'max'
        else:
            monitor_name, monitor_mode = 'mae', 'min'

        self.trainer = BaseTrainer(config=config.get('train', {}), logger=logger, num_gpus=num_gpus, sweep=sweep)
        
        # Visualization callbacks
        viz_cfg = config.get('viz') or config.get('advanced_visualization') or {}
        save_enabled = config.get('train', {}).get('save', False)
        from src.viz import create_viz_callbacks
        self.trainer.callbacks.extend(create_viz_callbacks(viz_cfg, save_enabled=save_enabled))
        
        # Preprocessor freeze callback
        warmup_cfg = config.get('warmup') or {}
        freeze_epochs = warmup_cfg.get('freeze_epochs', 0)
        if freeze_epochs != 0:
            self.trainer.callbacks.append(PreprocessorFreezeCallback(freeze_epochs=freeze_epochs))
        
        # Checkpoint callback: only when --save is set
        if save_enabled:
            has_wandb = logger is not None
            metric_key = f"val_{monitor_name}"
            filename_tmpl = '{epoch}-' + '{' + f'{metric_key}:.4f' + '}'
            
            if has_wandb:
                # With wandb: save to wandb artifacts (handled by log_model in WandbLogger)
                # Still need ModelCheckpoint for proper saving, but Lightning will sync to wandb
                checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(
                    dirpath=None,  # Let Lightning use default temp dir, wandb will upload
                    filename=filename_tmpl,
                    save_top_k=1,
                    monitor=metric_key,
                    mode=monitor_mode,
                    save_last=True,
                )
            else:
                # Without wandb: save to local CKPT_DIR
                os.makedirs(CKPT_DIR, exist_ok=True)
                checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(
                    dirpath=CKPT_DIR,
                    filename=filename_tmpl,
                    save_top_k=1,
                    monitor=metric_key,
                    mode=monitor_mode,
                    save_last=True,
                )
            self.trainer.callbacks.append(checkpoint_callback)
        
        # Early stopping
        earlystopping_callback = L.pytorch.callbacks.EarlyStopping(
            monitor=f'val_{monitor_name}',
            patience=patience,
            mode=monitor_mode,
            divergence_threshold=None,
            check_on_train_epoch_end=False,
            strict=False,
        )
        self.trainer.callbacks.append(earlystopping_callback)
        
        # Test trainer
        self.test_trainer = L.Trainer(
            devices=self.trainer.device0,
            accelerator=self.trainer.acc,
            logger=logger,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
        )

class Experiment:
    def __init__(self, config, use_wandb=False, num_gpus=None, sweep=False, ckpt_path=None, test_data=False):
        self.lightning_module = ViTLModule(config=config)
        self.data_module = ViTDataModule.from_config(config, test_data=test_data)
        self.lightning_module.sweep = sweep
        
        # Wandb logger setup
        if use_wandb:
            save_enabled = config.get('train', {}).get('save', False)
            run_name_suffix = config.get('wandb_run_suffix', '')
            run_name = f"{self.lightning_module.model.name}{run_name_suffix}"
            
            # log_model=True means save to wandb artifacts (only when save=True)
            logger = L.pytorch.loggers.WandbLogger(
                project=config['project'],
                config=config,
                name=run_name,
                log_model=save_enabled,  # Save to wandb artifacts if --save -w 1
                save_dir=SAVE_DIR if not sweep else SAVE_DIR
            )
        else:
            logger = None
        
        self.t = SpecTrainer(config=config, logger=logger, num_gpus=num_gpus, sweep=sweep)
        self.ckpt_path = ckpt_path
    
    def run(self):
        self.t.trainer.fit(self.lightning_module, datamodule=self.data_module, ckpt_path=self.ckpt_path)
        self.t.test_trainer.test(self.lightning_module, datamodule=self.data_module)
