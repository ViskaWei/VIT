import torch
from torch import nn, optim
from transformers import ViTConfig

import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from torchmetrics import Accuracy

from src.models import MyViT
from src.data import prepare_data

torch.manual_seed(42)

class LightningViTModel(L.LightningModule):
    def __init__(self, model, config, task):
        super().__init__()
        self.save_hyperparameters(ignore=[model])
        self.model = model
        self.config = config
        self.task = task     
        if task == 'cls':
        #     self.loss_fn = nn.CrossEntropyLoss()
            self.get_accuracy = Accuracy(task='multiclass', num_classes=config.num_labels)


    def forward(self, x, labels=None):
        return self.model(x, labels= labels)

    def training_step(self, batch, batch_idx):
        images, labels ,error,  = batch
        noise =  torch.randn(images.shape).to(self.device) * error if self.config.noise_level > 0 else 0 #point-wise multiplication 
        outputs = self(images + self.config.noise_level * noise, labels=labels)
        self.log('train_loss', outputs.loss, on_step=False,  on_epoch=True)
        return outputs.loss

    def validation_step(self, batch, batch_idx):
        images, labels ,error,  = batch
        noise =  torch.randn(images.shape).to(self.device) * error if self.config.noise_level > 0 else 0 #point-wise multiplication 
        outputs = self(images + self.config.noise_level * noise, labels=labels)
        self.log('val_loss', outputs.loss)
        if self.task == 'cls':
            accuracy = self.get_accuracy(outputs.logits, labels)
            self.log('val_acc', accuracy, on_step=False, on_epoch=True,  prog_bar=False)

    def test_step(self, batch, batch_idx):
        images, labels,error,  = batch
        outputs = self(images + self.config.noise_level * error, labels=labels)
        self.log('test_loss', outputs.loss, on_step=False, on_epoch=True)
        if self.task == 'cls':
            accuracy = self.get_accuracy(outputs.logits, labels)
            self.log('test_accuracy', accuracy, on_step=False, on_epoch=True)
        return outputs.loss

    def configure_optimizers(self):
        return optim.AdamW(self.parameters(), lr=self.config.learning_rate, weight_decay=1e-2)


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()

    # training params
    parser.add_argument("--exp-name", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--save-model-every", type=int, default=0)

    # model params
    parser.add_argument("--patch-size", type=int, default=100)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-hidden-layers", type=int, default=4)
    parser.add_argument("--num-attention-heads", type=int, default=4)
    parser.add_argument("--embed-fn", type=str, default="VPE")
    parser.add_argument("--proj-fn", type=str, default="C1D")
    parser.add_argument("--stride-ratio", type=float, default=1.0, help="ratio of stride in the sliding window")

    parser.add_argument("--task", type=str, default="cls")

    # data params
    parser.add_argument("--file-path", type=str, required=True)
    parser.add_argument("--num-samples", type=int, default=1000, help="Number of samples to load")
    parser.add_argument("--param-idx", type=int, default=1)
    parser.add_argument("--noise", type=float, default=0, help="noise level")

    # sys params
    parser.add_argument("--device", type=str, default=None, help="cpu or cuda or mps")
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs to use")
    parser.add_argument("--debug", action='store_true', help="Enable debug mode")
    parser.add_argument("--ckpt", type=str, default=None, help="resume from checkpoint")
    # wandb / saving
    parser.add_argument("--wandb-project", type=str, default="wandb-name-test", help="Wandb project name")
    parser.add_argument("--save", action='store_true', help="only when set, save checkpoints and log to W&B")

    args = parser.parse_args()

    def get_accelerator():
        args.num_device = 1
        if torch.cuda.is_available(): 
            args.num_devices = torch.cuda.device_count() if args.gpus == 0 else min(args.gpus, torch.cuda.device_count())
            return 'cuda'
        if torch.backends.mps.is_available(): return 'mps'
        args.gpus = 1
        return 'cpu'
    if args.device is None: args.device = get_accelerator()

    return args

def main():
    args = parse_args()

    train_dataloader, test_dataloader, num_classes, image_size = prepare_data(args)

    config = ViTConfig(
        image_size=image_size,
        patch_size=args.patch_size,
        num_channels=1,
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=args.num_attention_heads,
        intermediate_size=4 * args.hidden_size,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        is_encoder_decoder=False,
        use_mask_token=False,
        qkv_bias=True,
        num_labels=num_classes,
        noise_level = args.noise,
        learning_rate=args.lr,
        proj_fn = args.proj_fn,
        stride_ratio = args.stride_ratio
    )

    wandb_logger = None
    if (not args.debug) and args.save:
        import os
        wandb_logger = WandbLogger(project=args.wandb_project, log_model=True, config=vars(args), save_dir=os.environ.get('WANDB_DIR', './wandb'))

    model = MyViT(config)
    lightning_model = LightningViTModel(model, config, args.task)

    # Define callbacks
    callbacks = []
    if args.save:
        checkpoint_callback = ModelCheckpoint(
            dirpath=f'checkpoints/{args.exp_name}',
            filename='{epoch}-{val_loss:.2f}',
            save_top_k=3,
            monitor='val_loss',
            mode='min',
            every_n_epochs=args.save_model_every if args.save_model_every > 0 else 1
        )
        callbacks.append(checkpoint_callback)
    # Initialize the PyTorch Lightning Trainer
    trainer = L.Trainer(
        max_epochs=args.epochs,
        accelerator=args.device,
        devices=args.num_device,
        strategy='ddp' if args.gpus > 1 else 'auto',  # Use DistributedDataParallel for multi-GPU
        callbacks=callbacks,
        default_root_dir=f'experiments/{args.exp_name}',
        fast_dev_run=args.debug,
        logger=wandb_logger if wandb_logger is not None else False,
    )
    # No logger when --save is not set
    trainer.fit(lightning_model, train_dataloader, train_dataloader, ckpt_path=args.ckpt)  # Using same dataloader for train and val
    trainer.test(lightning_model, test_dataloader)
    
if __name__ == "__main__":
    main()
