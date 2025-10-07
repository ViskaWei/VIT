import os
import torch
import torch.nn as nn
import lightning as L
from torch.utils.data import DataLoader, TensorDataset
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Type

from src.dataloader import (
    BaseDataset,
    BaseSpecDataset,
    Configurable,
    MaskMixin,
    NoiseMixin,
    SingleSpectrumNoiseDataset,
)

#region DATA-----------------------------------------------------------
# NOTE: Dataset classes now live under src.dataloader.* and are imported above.
#endregion DATA-----------------------------------------------------------
#region DM-----------------------------------------------------------
class BaseDataModule(L.LightningDataModule):
    def __init__(self, batch_size: int = 256, num_workers: int = 24, debug: bool = False, dataset_cls: Optional[Type['BaseDataset']] = None, config: Dict[str, Any] = {}):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.debug = debug
        self.dataset_cls = dataset_cls
        self.config = config
        self.reload = True

    @classmethod
    def from_config(cls, dataset_cls=BaseDataset, config: Dict[str, Any]={}):
        train_config = config.get('train', {})
        # Accept both 'num_workers' and legacy 'workers'
        num_workers = train_config.get('num_workers', train_config.get('workers', 24))
        return cls(
            batch_size=train_config.get('batch_size', 256),
            num_workers=num_workers,
            debug=train_config.get('debug', False),
            dataset_cls=dataset_cls,
            config=config,
        )

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train = self.dataset_cls.from_config(self.config)
            self.train.load_data()
            self.val = self.setup_test_dataset(stage='val')
            self.val.load_data(stage='val')
        elif stage == 'test':
            if self.reload:
                self.test = self.setup_test_dataset(stage)
                self.test.load_data(stage=stage)
            
    def setup_test_dataset(self):
        test_data, test_params = self.train.get_testdata()
        self.test_dict.update(test_params)
        return TensorDataset(*test_data)

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            shuffle=not self.debug
            # shuffle=False,
        )
    
    def val_dataloader(self):
        # Use a sane validation batch size instead of the entire dataset.
        # Large datasets (e.g., 100k) will OOM or hit CUDA kernel limits if batched as one.
        val_bs = min(self.batch_size if self.batch_size and self.batch_size > 0 else 1, len(self.val))
        return DataLoader(
            self.val,
            batch_size=val_bs,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            shuffle=False,
        )
    
    def test_dataloader(self):
        return DataLoader(self.test,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          pin_memory=True,
                          persistent_workers=True,
                          shuffle=False) 
#endregion DM-----------------------------------------------------------

#region MODEL-----------------------------------------------------------
class BaseModel(nn.Module, ABC):
    def __init__(self, model_name='model', loss_name='train'):
        super(BaseModel, self).__init__()
        self._model_name = model_name
        self._loss_name = loss_name
        print(f'Creating {self._model_name} model with {self._loss_name} loss')
        
    @property
    def name(self):
        return self._model_name
    @property
    def loss_name(self):
        return self._loss_name
    
    @abstractmethod
    def forward(self, x, labels=None):
        pass
    @abstractmethod
    def compute_loss(self, *args, **kwargs):
        pass
    @abstractmethod
    def log_outputs(self, outputs, log_fn = print, stage=''):
        log_fn(f'{self.loss_name}_loss', outputs['loss'])


#endregion MODEL-----------------------------------------------------------

class BaseLightningModule(L.LightningModule):
    def __init__(self, model: BaseModel = None,config={}):
        super().__init__()
        self.model = model
        self.loss_name = model.loss_name 
        self.callbacks = []
        self.config = config
        self.sweep = False

    def configure_optimizers(self):
        opt_config = {**self.config.get('opt', {}), 'loss_name': self.loss_name}
        return OptModule.from_config(opt_config)(self.model)
    
    def on_fit_start(self):
        
        if self.logger and hasattr(self.logger, 'experiment') and hasattr(self.logger.experiment, 'log'):
            num_params = sum(p.numel() for p in self.model.parameters())
            self.logger.experiment.log({"num_params(M)": round(num_params / 1e6, 3)})
            
    def on_train_epoch_start(self):
        """Log the current learning rate at the start of each epoch."""
        optimizer = self.trainer.optimizers[0]  # Assuming one optimizer
        lr = optimizer.param_groups[0]['lr']
        
        if self.logger and hasattr(self.logger, 'experiment') and hasattr(self.logger.experiment, 'log'):
            self.logger.experiment.log({"lr": lr})






class BaseTrainer(L.Trainer):
    def __init__(self, config, logger=False, num_gpus=None, sweep=False):
        # Saving behavior: only save when config['save'] is truthy
        enable_checkpointing = bool(config.get('save', False))
        # UI behavior can still depend on sweep mode
        if sweep:
            num_gpus = 1
            enable_progress_bar = False
            enable_model_summary = False
        else:
            enable_progress_bar = True
            enable_model_summary = True
        self.acc, self.device0 = self.select_device(num_gpus or config.get('gpus'))
        epoch = config.get('ep', 10)
        # Allow precision override via config.train.precision, default FP32
        precision = str(config.get('precision', '32'))
        super().__init__(
            max_epochs=epoch,
            devices=self.device0,
            accelerator=self.acc,
            strategy='ddp' if self.device0 and self.device0 > 1 else 'auto',
            logger=logger,
            precision=precision,
            gradient_clip_val=config.get('grad_clip', 0.5),
            # stochastic_weight_avg=True,
            fast_dev_run=config.get('debug', False),
            enable_checkpointing=enable_checkpointing,
            enable_progress_bar=enable_progress_bar,
            enable_model_summary=enable_model_summary,
        )
    def select_device(self, num_gpus: Optional[int] = None):
        """Return accelerator type and device count based on availability."""
        if num_gpus and num_gpus > 0:
            if torch.cuda.is_available():
                return 'gpu', num_gpus
            if torch.backends.mps.is_available():
                return 'mps', 1
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            return 'gpu', torch.cuda.device_count()
        if torch.backends.mps.is_available():
            return 'mps', 1
        return 'cpu', 1
    
        #     enable_checkpointing = True
        #     enable_progress_bar = True
        #     enable_model_summary = True
        #     num_gpus = num_gpus or config.get('gpus', None) or torch.cuda.device_count()
        # epoch = config.get('ep', 10)
        # super().__init__(
        #     max_epochs=epoch,
        #     devices=num_gpus,
        #     accelerator='gpu',
        #     strategy='ddp' if num_gpus > 1 else 'auto',
        #     logger = logger,
        #     precision = '32',
        #     gradient_clip_val=config.get('grad_clip', 0.5),
        #     # stochastic_weight_avg=True,
        #     fast_dev_run=config.get('debug', False),
        #     enable_checkpointing=enable_checkpointing,
        #     enable_progress_bar=enable_progress_bar,
        #     enable_model_summary=enable_model_summary
        # )
        
        
class PlotCallback(L.pytorch.callbacks.Callback):
    def __init__(self, interval=2, plotter=None, folder_name='plot', root_dir='./'):
        super().__init__()
        self.interval = interval
        self.p = plotter
        self.folder_name = folder_name
        self.dir = os.path.join(root_dir, self.folder_name)
        os.makedirs(self.dir, exist_ok=True)
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]):
        return cls(**{k: v for k, v in config.items() if k in cls.__init__.__code__.co_varnames})

    def on_train_epoch_end(self, trainer, pl_module):
        if (trainer.current_epoch + 1) % self.plot_interval == 0:
            img_dict = self.plot_epoch_results(trainer, pl_module)
            for key, fig in img_dict.items():
                self.save_plot(fig, key, logger=None)

    def on_fit_end(self, trainer, pl_module):
        if trainer.logger and hasattr(trainer.logger, 'log_image'):
            self.logger = trainer.logger

        # self.create_gif()

    def plot_epoch_results(self, trainer, pl_module):
        pl_module.eval()
        with torch.no_grad():
            # Assuming your test data is stored in pl_module.test_dataset
            test_dataloader = pl_module.test_dataloader()
            batch = next(iter(test_dataloader))
            output_dict = pl_module.test_step(batch)
            self.p.plot(output_dict)
            img_dict = pl_module.model.plot(batch, output_dict,)
        return img_dict

    def save_plot(self, fig, name, logger = None):
        if logger is None:
            fig.savefig(os.path.join(self.dir, f'{name}.png'))
        else:
            logger.log_image(name, fig)

class OptModule():
    def __init__(self, lr, monitor_name='train_loss', opt_type='adam', weight_decay=0.0, lr_scheduler_name=None, **kwargs) -> None:
        self.lr = float(lr)
        self.monitor_name=monitor_name
        self.opt_type = opt_type
        self.weight_decay = weight_decay
        self.lr_scheduler_name = lr_scheduler_name
        self.kwargs = kwargs
        self.opt_fns = {
            'adam': torch.optim.Adam,
            'adamw': torch.optim.AdamW,
            'sgd': torch.optim.SGD,
            'rmsprop': torch.optim.RMSprop,
            'adadelta': torch.optim.Adadelta,
            'adagrad': torch.optim.Adagrad,
            'adamax': torch.optim.Adamax,
            'asgd': torch.optim.ASGD,
            'lbfgs': torch.optim.LBFGS,
            'rprop': torch.optim.Rprop,
            'sparseadam': torch.optim.SparseAdam,
        }
        self.lr_schedulers = {
            'cosine': torch.optim.lr_scheduler.CosineAnnealingLR,
            'cosineannealing': torch.optim.lr_scheduler.CosineAnnealingLR,
            'cosineannealinglr': torch.optim.lr_scheduler.CosineAnnealingLR,
            'onecycle': torch.optim.lr_scheduler.OneCycleLR,
            'plateau': torch.optim.lr_scheduler.ReduceLROnPlateau,
        }

    @classmethod
    def from_config(cls, config):
        """
        Create an OptModule instance from a configuration dictionary.
        
        Args:
            config (dict): A dictionary containing the configuration parameters.
                           Expected keys: 'lr', 'type' (optional), 'lr_scheduler' (optional)
        
        Returns:
            OptModule: An instance of the OptModule class
        """
        lr = config.get('lr', 1e-3)
        opt_type = config.get('type', 'adam').lower()  # Convert to lowercase
        weight_decay = config.get('weight_decay', 0)
        loss_name = config.get('loss_name', 'train')
        # Align with validation logging in ViTLModule: 'val_{loss_name}_loss'
        monitor_name = f"val_{loss_name}_loss"
        if 'lr_sch' in config:
            lr_scheduler_name = config['lr_sch'].lower()  # Convert to lowercase
            # Filter out scheduler-specific kwargs - only keep relevant ones based on scheduler type
            exclude_keys = {'lr', 'type', 'lr_sch', 'weight_decay', 'loss_name', 'factor', 'patience'}
            kwargs = {k: v for k, v in config.items() if k not in exclude_keys}
            # Add back specific params if needed by the scheduler
            if 'cosineannealing' in lr_scheduler_name:
                # CosineAnnealingLR needs T_max
                if 'T_max' in config:
                    kwargs['T_max'] = config['T_max']
                elif 'ep' in config.get('train', {}):
                    kwargs['T_max'] = config['train']['ep']
            elif 'plateau' in lr_scheduler_name:
                # ReduceLROnPlateau needs factor, patience
                if 'factor' in config:
                    kwargs['factor'] = config['factor']
                if 'patience' in config:
                    kwargs['patience'] = config['patience']
            return cls(lr=lr, monitor_name=monitor_name, opt_type=opt_type, weight_decay=weight_decay, lr_scheduler_name=lr_scheduler_name, **kwargs)
        return cls(lr=lr, monitor_name=monitor_name, opt_type=opt_type, weight_decay=weight_decay)

    def __call__(self, model):
        optimizer = self.opt_fns[self.opt_type](model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        
        if self.lr_scheduler_name is None:
            return optimizer
        if self.lr_scheduler_name not in self.lr_schedulers:
            raise ValueError(f"Unknown scheduler: {self.lr_scheduler_name}")
        scheduler = self.lr_schedulers[self.lr_scheduler_name](optimizer, **self.kwargs)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, 'monitor': f'{self.monitor_name}'}


__all__ = [
    "Configurable",
    "BaseDataset",
    "MaskMixin",
    "NoiseMixin",
    "SingleSpectrumNoiseDataset",
    "BaseSpecDataset",
    "BaseDataModule",
    "BaseModel",
    "BaseLightningModule",
    "BaseTrainer",
    "PlotCallback",
    "OptModule",
]
