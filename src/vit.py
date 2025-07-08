import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchmetrics import Accuracy

from src.basemodule import BaseLightningModule, BaseTrainer, BaseSpecDataset, BaseDataModule

SAVE_DIR='/datascope/subaru/user/swei20/wandb'
SAVE_PATH = '/home/swei20/SirenSpec/checkpoints'
MASK_PATH = '/datascope/subaru/user/swei20/model/bosz50000_mask.npy'

#region --DATA-----------------------------------------------------------
class ClassSpecDataset(BaseSpecDataset):
    def __init__(self, param_idx=1, **kwargs):
        super().__init__(**kwargs)
        self.param_idx = param_idx  
        
    def load_data(self, stage=None):
        super().load_data(stage)
        self.load_params(stage)
        self.labels = (torch.tensor(self.logg > 2.5)).long() 

    def __getitem__(self, idx):
        flux, error = super().__getitem__(idx)
        return flux, error, self.labels[idx] 
    
#endregion --DATA-----------------------------------------------------------
#region --DATAMODULE-----------------------------------------------------------
class ViTDataModule(BaseDataModule):
    @classmethod
    def from_config(cls, config):
        return super().from_config(dataset_cls=ClassSpecDataset, config=config)
    def setup_test_dataset(self, stage):
        return ClassSpecDataset.from_config(self.config)
#endregion --DATAMODULE-----------------------------------------------------------

#region MODEL-----------------------------------------------------------
from transformers import ViTModel, ViTConfig
from src.model import MyViT
def get_vit_model(config):
    """
    Create a Vision Transformer model based on the provided configuration.
    Args:
        config (dict): Configuration dictionary containing model parameters.
        num_classes (int): Number of output classes for classification tasks.
    Returns:
        MyViT: Instance of the Vision Transformer model.
    """
    vit_config = get_model_config(config)
    # Initialize the model with the ViTConfig
    return MyViT(vit_config)

def get_model_config(config):
    """
    Create a ViTConfig object based on the provided configuration.
    Args:
        config (dict): Configuration dictionary containing model parameters.
        num_classes (int): Number of output classes for classification tasks.
        image_size (int): Size of the input images.
    Returns:
        ViTConfig: Config object for the Vision Transformer model.
    """
    
    vit_config = ViTConfig(
        image_size=config['model']['image_size'],
        patch_size=config['model']['patch_size'],
        num_channels=1,
        hidden_size=config['model']['hidden_size'],
        num_hidden_layers=config['model']['num_hidden_layers'],
        num_attention_heads=config['model']['num_attention_heads'],
        intermediate_size=4 * config['model']['hidden_size'],
        stride_ratio=config['model']['stride_ratio'],
        proj_fn=config['model']['proj_fn'],

        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        is_encoder_decoder=False,
        use_mask_token=False,
        qkv_bias=True,
        num_labels=config['model']['num_labels'],
        # noise_level=config['noise']['noise_level'],
        # learning_rate=config['opt']['lr'],
    )
    return vit_config


#region --TRAINER-----------------------------------------------------------
class ViTLModule(BaseLightningModule):
    def __init__(self, model=None, config = {}):
        model = model or self.get_model(config)
        super().__init__(model=model, config=config)
        self.save_hyperparameters()
        self.loss_name = 'train'  # Set the loss name for logging
        self.model.loss_name = self.loss_name  # Ensure the model has the loss name set
        self.get_accuracy = Accuracy(task='multiclass', num_classes=config['model']['num_labels'])

    def get_model(self, config):
        return get_vit_model(config)

    def forward(self, flux, labels, loss_only=True):
        outputs = self.model(flux, labels=labels)
        if loss_only:
            return outputs.loss
        else:
            return outputs
        
    def training_step(self, batch, batch_idx):
        flux, _, labels = batch
        loss = self.forward(flux, labels, loss_only=True)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        flux, _, labels = batch
        outputs = self.forward(flux, labels, loss_only=False)
        self.log('val_loss', outputs.loss, on_step=True, on_epoch=True, prog_bar=True)
        accuracy = self.get_accuracy(outputs.logits, labels)
        self.log('val_acc', accuracy, on_step=False, on_epoch=True,  prog_bar=False)
        return outputs.loss

    def test_step(self, batch, batch_idx):
        flux, _, labels = batch
        outputs = self.forward(flux, labels, loss_only=False)
        self.log('test_loss', outputs.loss, on_step=True, on_epoch=True, prog_bar=True)
        accuracy = self.get_accuracy(outputs.logits, labels)
        self.log('test_acc', accuracy, on_step=False, on_epoch=True,  prog_bar=False)
        return outputs.loss
 
#endregion --TRAINER-----------------------------------------------------------

import lightning as L

class SpecTrainer():
    def __init__(self, config, logger, num_gpus=None, sweep=False, monitor_name='acc', monitor_mode='max') -> None:
        if sweep: num_gpus = 1
        patience = 100 if sweep else 500
        self.trainer = BaseTrainer(config=config.get('train', {}), logger=logger, num_gpus=num_gpus, sweep=sweep)  #
        if not sweep: 
            checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(dirpath= SAVE_PATH, filename='{epoch}-{acc_valid:.0f}', save_top_k=1, monitor=f'val_{monitor_name}', mode=monitor_mode)
            self.trainer.callbacks.append(checkpoint_callback)
            
        earlystopping_callback = L.pytorch.callbacks.EarlyStopping(monitor=f'val_{monitor_name}', patience=patience, mode=monitor_mode, divergence_threshold=1,)
        self.trainer.callbacks.append(earlystopping_callback)
        self.test_trainer = L.Trainer(devices=1, accelerator='gpu', logger=logger,  enable_checkpointing=False, enable_progress_bar=False, enable_model_summary=False)

class Experiment:
    def __init__(self, config, use_wandb=False, num_gpus=None, sweep=False, ckpt_path=None):
        # self.lightning_module = BlindspotLModule(config=config)
        self.lightning_module = ViTLModule(config=config)
        self.data_module = ViTDataModule.from_config(config)

        self.lightning_module.sweep = sweep
        if use_wandb:
            if sweep:
                logger = L.pytorch.loggers.WandbLogger(config=config, name=self.lightning_module.model.name, log_model=False, save_dir=SAVE_DIR) 
            else:
                logger = L.pytorch.loggers.WandbLogger(project = config['project'], config=config, name=self.lightning_module.model.name, log_model=True, save_dir=SAVE_DIR)
        else:
            logger = None
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
    import yaml
    def load_config(config_path):
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config

# /home/swei20/VIT/configs/vit.yaml
    config  = load_config('./configs/vit.yaml')
    exp = Experiment(config, use_wandb=True, num_gpus=1)
    exp.run()
    print('Experiment completed successfully!')