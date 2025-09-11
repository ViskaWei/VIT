import os
import torch
from torchmetrics import Accuracy, MeanAbsoluteError, R2Score
import lightning as L

from src.basemodule import BaseLightningModule, BaseTrainer, BaseSpecDataset, BaseDataModule
# from src.callbacks_pca_warm import PCAWarmStartCallback, CKAProbeCallback

from src.utils import make_dummy_spectra


# Use env override for local W&B run files
# Falls back to ./wandb in the project if not set
SAVE_DIR = os.environ.get('WANDB_DIR', './wandb')
SAVE_PATH = './checkpoints'
# MASK_PATH = './bosz50000_mask.npy'

#region --DATA-----------------------------------------------------------
def _normalize_task(config):
    """Return ('cls'|'reg') from config supporting legacy keys."""
    m = (config.get('model', {}) or {})
    task = (m.get('task_type') or m.get('task') or 'cls').lower()
    if task in ('classification', 'cls', 'class'):
        return 'cls'
    return 'reg'
class TestDataset(BaseSpecDataset):
    def __init__(self, task='classification', **kwargs):
        super().__init__(**kwargs)
        self.task = task

    @classmethod
    def from_config(cls, config):
        d = super().from_config(config)
        d.task = 'regression' if _normalize_task(config) == 'reg' else 'classification'
        return d
        
    def load_data(self, stage=None):
        spectra = make_dummy_spectra(self.num_samples, 4096)
        self.flux = spectra
        # Provide a dummy error tensor so DataLoader collate doesn't see None
        self.error = torch.zeros_like(self.flux) + 1e-3
        if self.task == 'regression':
            # Dummy continuous targets
            self.labels = torch.randn(spectra.shape[0])
        else:
            self.labels = torch.randint(0, 2, (spectra.shape[0],)).long()

    def __getitem__(self, idx):
        return self.flux[idx], self.error[idx], self.labels[idx]
    
class ClassSpecDataset(BaseSpecDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @classmethod
    def from_config(cls, config):
        d = super().from_config(config)
        return d
        
    def load_data(self, stage=None):
        super().load_data(stage)
        self.load_params(stage)
        self.labels = (torch.tensor(self.logg > 2.5)).long()

    def __getitem__(self, idx):
        flux, error = super().__getitem__(idx)
        return flux, error, self.labels[idx]

class RegSpecDataset(BaseSpecDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Stats used for label normalization (if enabled)
        self.label_mean = None
        self.label_std = None
        self.label_min = None
        self.label_max = None

    @classmethod
    def from_config(cls, config):
        d = super().from_config(config)
        return d

    def load_data(self, stage=None):
        super().load_data(stage)
        self.load_params(stage)
        # Enforce explicit `data.param` for regression targets
        if not (
            (isinstance(getattr(self, 'param', None), str) and len(self.param) > 0)
            or (isinstance(getattr(self, 'param', None), (list, tuple)) and len(self.param) > 0)
        ):
            raise ValueError("Regression requires 'data.param' to be set in the config (string, comma-separated string, or list).")
        self.labels = torch.tensor(self.param_values).float()  # shape: (N,) or (N,K)
        # Optional label normalization for regression
        self._maybe_normalize_labels(stage)

    def __getitem__(self, idx):
        flux, error = super().__getitem__(idx)
        return flux, error, self.labels[idx]

    def _maybe_normalize_labels(self, stage=None,  kind=None, eps = 1e-8):
        kind = getattr(self, 'label_norm', 'none')
        if kind not in ('standard', 'zscore', 'minmax'):
            return
        is_train = stage in (None, 'fit', 'train')
        # Compute along batch dim; for multi-output K>1, stats are 1D tensors of length K
        if kind in ('standard', 'zscore'):
            if is_train or (self.label_mean is None or self.label_std is None):
                self.label_mean = self.labels.mean(dim=0, keepdim=False)
                self.label_std = self.labels.std(dim=0, unbiased=False, keepdim=False)
            std = self.label_std.clone() if isinstance(self.label_std, torch.Tensor) else torch.tensor(self.label_std)
            std = torch.where(std.abs() < eps, torch.ones_like(std), std)
            self.labels = (self.labels - self.label_mean) / std
        elif kind == 'minmax':
            if is_train or (self.label_min is None or self.label_max is None):
                self.label_min = self.labels.min(dim=0, keepdim=False).values
                self.label_max = self.labels.max(dim=0, keepdim=False).values
            denom = (self.label_max - self.label_min)
            denom = torch.where(denom.abs() < eps, torch.ones_like(denom), denom)
            self.labels = (self.labels - self.label_min) / denom
        # Lightweight log of scalar summaries for visibility
        try:
            m = self.label_mean if isinstance(self.label_mean, torch.Tensor) else None
            s = self.label_std if isinstance(self.label_std, torch.Tensor) else None
            mi = self.label_min if isinstance(self.label_min, torch.Tensor) else None
            ma = self.label_max if isinstance(self.label_max, torch.Tensor) else None
            def _fmt(x):
                if x is None: return None
                x = x.detach().cpu().flatten()
                return [round(float(v), 4) for v in x.tolist()[:4]]  # show up to first 4 dims
            print(f"[{stage or 'all'} data] label normalization '{kind}': mean={_fmt(m)}, std={_fmt(s)}, min={_fmt(mi)}, max={_fmt(ma)}")
        except Exception:
            pass
    
#endregion --DATA-----------------------------------------------------------
#region --DATAMODULE-----------------------------------------------------------
class ViTDataModule(BaseDataModule):
    @classmethod
    def from_config(cls, config, test_data=False):
        task = _normalize_task(config)
        if test_data:
            dataset_cls = TestDataset
            print('Using Test Dataset')
        else:
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
from src.model import get_model

#region --TRAINER-----------------------------------------------------------
class ViTLModule(BaseLightningModule):
    def __init__(self, model=None, config = {}):
        model = model or self.get_model(config)
        super().__init__(model=model, config=config)
        self.save_hyperparameters(ignore=['model'])
        self.task_type = _normalize_task(config)
        if self.task_type == 'cls':
            self.accuracy = Accuracy(task='multiclass', num_classes=config['model']['num_labels'])
        elif self.task_type == 'reg':
            self.mae = MeanAbsoluteError()
            self.r2 = R2Score()

    def get_model(self, config):
        return get_model(config)

    def forward(self, flux, labels, loss_only=True):
        """Forward wrapper returning loss or full outputs from HF model."""
        outputs = self.model(flux,labels=labels)
        return outputs.loss if loss_only else outputs
        
    def training_step(self, batch, batch_idx):
        flux, _, labels = batch
        loss = self.forward(flux, labels, loss_only=True)
        self.log(f'{self.loss_name}_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    def _shared_eval_step(self, batch, prefix):
        flux, _, labels = batch
        outputs = self.forward(flux, labels, loss_only=False)
        loss = outputs.loss
        self.log(f'{prefix}_{self.loss_name}_loss', loss, on_step=False, on_epoch=True)
        if self.task_type == 'cls':
            acc = self.accuracy(outputs.logits, labels)
            self.log(f'{prefix}_acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        elif self.task_type == 'reg':
            preds = outputs.logits.squeeze()
            mae = self.mae(preds, labels)
            r2 = self.r2(preds, labels)
            self.log(f'{prefix}_mae', mae, on_step=False, on_epoch=True)
            self.log(f'{prefix}_r2', r2, on_step=False, on_epoch=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        return self._shared_eval_step(batch, 'val')

    def test_step(self, batch, batch_idx):
        return self._shared_eval_step(batch, 'test')

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
            monitor_name, monitor_mode, filename_suffix = 'acc', 'max', '{acc_valid:.0f}'
        else:
            monitor_name, monitor_mode, filename_suffix = 'mae', 'min', '{mae:.2f}'

        self.trainer = BaseTrainer(config=config.get('train', {}), logger=logger, num_gpus=num_gpus, sweep=sweep)
        # p = (config.get('pca') or {})
        # if p.get('warm', False):
        #     self.trainer.callbacks.append(PCAWarmStartCallback(
        #         attn_module_path=p.get('attn_path', 'model.vit.encoder.layer.0.attention'),
        #         r=int(p.get('r', 32)),
        #         robust=bool(p.get('robust', False)),
        #         kernel=p.get('kernel', None),
        #         nystrom_m=int(p.get('nystrom_m', 256)),
        #         whiten=bool(p.get('whiten', False)),
        #         trigger_epoch=0,
        #     ))
        # if p.get('cka', False):
        #     self.trainer.callbacks.append(CKAProbeCallback(
        #         attn_module_path=p.get('attn_path', 'model.vit.encoder.layer.0.attention'),
        #         r=int(p.get('r', 32)),
        #         every_n_epochs=int(p.get('cka_every', 1)),
        #         kernel=p.get('cka_kernel', 'linear'),
        #     ))
        
        # Add checkpointing only when saving is enabled
        if (config.get('train', {}).get('save', False)):
            checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(dirpath= SAVE_PATH, filename='{epoch}-{acc_valid:.0f}', save_top_k=1, monitor=f'val_{monitor_name}', mode=monitor_mode)
            self.trainer.callbacks.append(checkpoint_callback)
            
        earlystopping_callback = L.pytorch.callbacks.EarlyStopping(monitor=f'val_{monitor_name}', patience=patience, mode=monitor_mode, divergence_threshold=1,)
        self.trainer.callbacks.append(earlystopping_callback)
        # For regression, add original-scale plotting callback at test time
        # if task_type == 'reg':
        #     self.trainer.callbacks.append(OrigScalePredPlotCallback(save_dir=SAVE_PATH))
        self.test_trainer = L.Trainer(
            devices=self.trainer.device0,
            accelerator=self.trainer.acc,
            logger=logger,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
        )
        # self.test_trainer = L.Trainer(devices=1, accelerator='gpu', logger=logger,  enable_checkpointing=False, enable_progress_bar=False, enable_model_summary=False)

class Experiment:
    def __init__(self, config, use_wandb=False, num_gpus=None, sweep=False, ckpt_path=None, test_data=False):
        # self.lightning_module = BlindspotLModule(config=config)
        self.lightning_module = ViTLModule(config=config)
        self.data_module = ViTDataModule.from_config(config, test_data=test_data)

        self.lightning_module.sweep = sweep
        if use_wandb:
            log_model =  config.get('train', {}).get('save', False) 
            if sweep:
                logger = L.pytorch.loggers.WandbLogger(config=config, name=self.lightning_module.model.name, log_model=False, save_dir=SAVE_DIR) 
            else:
                logger = L.pytorch.loggers.WandbLogger(project = config['project'], config=config, name=self.lightning_module.model.name, log_model=log_model, save_dir=SAVE_DIR)
        else:
            logger = None
        # Choose monitor based on task
        self.t = SpecTrainer(config = config, logger = logger, num_gpus=num_gpus, sweep=sweep)
        self.ckpt_path = ckpt_path
    
    def run(self):
        self.t.trainer.fit(self.lightning_module, datamodule=self.data_module, ckpt_path=self.ckpt_path)
        self.t.test_trainer.test(self.lightning_module, datamodule=self.data_module)
    
if __name__ == '__main__':
    # config = {
    #     'loss': {'name': 'E1'},
    #     'data': {'file_path': './tests/spec/test_dataset.h5', 'num_samples': 10,},
    #     'mask': {'mask_ratio': 0.9, },
    #     'noise': {'noise_level': 2.0, },
    #     'train': {'ep': 2},
    #     'model': {'input_sigma': True, 'blindspot': True, 'num_layers': 3, 'embed_dim': 3, 'kernel_size': 3}
    # }
    from src.utils import load_config

# /home/swei20/VIT/configs/vit.yaml
    config  = load_config('./configs/vit.yaml')
    exp = Experiment(config, use_wandb=True, num_gpus=1, test_data=True)
    exp.run()
    print('Experiment completed successfully!')
