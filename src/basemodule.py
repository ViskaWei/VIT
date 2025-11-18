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
from src.opt.optimizer import OptModule
from src.hardware_utils import (
    get_num_workers_from_config,
    select_accelerator_and_devices,
    get_training_strategy,
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
        """Create DataModule from config with smart worker auto-detection.
        
        Worker detection priority:
        1. Environment variable NUM_WORKERS (highest)
        2. Config train.num_workers or train.workers
        3. Auto-detection based on system resources (see src.worker_utils)
        """
        train_config = config.get('train', {})
        
        # Use worker_utils for clean, centralized worker detection
        num_workers, batch_size = get_num_workers_from_config(config, verbose=True)
            
        return cls(
            batch_size=batch_size,
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
        if not hasattr(self, 'val') or self.val is None:
            print("[WARNING] Validation dataset is None - validation will be skipped")
            return None
        if len(self.val) == 0:
            print("[WARNING] Validation dataset is empty - validation will be skipped")
            return None
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
                          persistent_workers=self.num_workers > 0,
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
        opt_config = {**self.config.get('opt', {})}
        # Use monitor_metric if available (e.g., 'mae' for regression), otherwise fall back to loss_name
        monitor_metric = getattr(self, 'monitor_metric', self.loss_name)
        opt_config['monitor_metric'] = monitor_metric
        
        # Check if validation data path is configured (required for ReduceLROnPlateau)
        lr_sch = opt_config.get('lr_sch', '').lower()
        data_config = self.config.get('data', {})
        has_val_path = bool(data_config.get('val_path'))
        
        if 'plateau' in lr_sch and not has_val_path:
            print(f"[WARNING] ReduceLROnPlateau requires validation data ('data.val_path' in config) but none configured.")
            print(f"[WARNING] Disabling learning rate scheduler. Consider adding validation data or using a different scheduler.")
            opt_config.pop('lr_sch', None)
        
        # For OneCycleLR, we need to inject steps_per_epoch and epochs
        if 'onecycle' in lr_sch:
            # Get training parameters
            train_config = self.config.get('train', {})
            batch_size = train_config.get('batch_size', 64)
            num_samples = data_config.get('num_samples', 32000)
            epochs = train_config.get('ep', 100)
            
            # Calculate steps per epoch
            steps_per_epoch = (num_samples + batch_size - 1) // batch_size
            opt_config['steps_per_epoch'] = steps_per_epoch
            opt_config['epochs'] = epochs
            print(f"[OneCycleLR] Calculated steps_per_epoch={steps_per_epoch}, epochs={epochs}")
        
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
        """Initialize PyTorch Lightning Trainer with smart hardware detection.
        
        Args:
            config: Training configuration dict
            logger: Optional logger instance
            num_gpus: Number of GPUs to use (None = auto-detect)
            sweep: Whether running in sweep mode (affects UI)
        """
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
        
        # Use hardware_utils for clean device detection
        self.acc, self.device0 = select_accelerator_and_devices(
            num_gpus or config.get('gpus')
        )
        strategy = get_training_strategy(self.device0)
        
        epoch = config.get('ep', 10)
        # Allow precision override via config.train.precision, default FP32
        precision = str(config.get('precision', '32'))
        
        # Ensure validation runs every epoch (required for ReduceLROnPlateau)
        # check_val_every_n_epoch=1 ensures metrics are available for scheduler
        super().__init__(
            max_epochs=epoch,
            devices=self.device0,
            accelerator=self.acc,
            strategy=strategy,
            logger=logger,
            precision=precision,
            gradient_clip_val=config.get('grad_clip', 0.5),
            fast_dev_run=bool(config.get('debug', False)),
            enable_checkpointing=enable_checkpointing,
            enable_progress_bar=enable_progress_bar,
            enable_model_summary=enable_model_summary,
            check_val_every_n_epoch=1,  # Run validation every epoch
            deterministic=True,  # Ensure reproducibility
        )

#endregion TRAINER------------------------------------------------------------
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
