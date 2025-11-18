import torch
from typing import Dict, Any

class OptModule():
    def __init__(self, lr, monitor_metric='loss', opt_type='adam', weight_decay=0.0, lr_scheduler_name=None, warmup_ratio=0.0, warmup_epochs=None, **kwargs) -> None:
        self.lr = float(lr)
        self.monitor_metric = monitor_metric
        self.opt_type = opt_type
        self.weight_decay = weight_decay
        self.lr_scheduler_name = lr_scheduler_name
        self.warmup_ratio = warmup_ratio
        self.warmup_epochs = warmup_epochs
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
            'constant': torch.optim.lr_scheduler.ConstantLR,
            'constantlr': torch.optim.lr_scheduler.ConstantLR,
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
        monitor_metric = config.get('monitor_metric', 'loss')
        
        # Warmup configuration from config.warmup or config directly
        warmup_config = config.get('warmup', {})
        warmup_ratio = warmup_config.get('ratio', config.get('warmup_ratio', 0.0))
        warmup_epochs = warmup_config.get('epochs', config.get('warmup_epochs', None))
        
        if 'lr_sch' in config:
            lr_scheduler_name = config['lr_sch'].lower()  # Convert to lowercase
            # Base exclude keys
            exclude_keys = {'lr', 'type', 'lr_sch', 'weight_decay', 'monitor_metric', 'loss_name'}
            kwargs = {}
            
            # Scheduler-specific parameter handling
            if 'cosine' in lr_scheduler_name:
                # CosineAnnealingLR needs T_max
                kwargs['T_max'] = config.get('T_max', config.get('ep', 100))
                # Optional: eta_min for minimum LR
                if 'eta_min' in config:
                    kwargs['eta_min'] = config['eta_min']
                    
            elif 'onecycle' in lr_scheduler_name:
                # OneCycleLR needs total_steps or (steps_per_epoch + epochs)
                # These will be calculated in configure_optimizers
                kwargs['max_lr'] = lr  # OneCycle's max_lr
                if 'steps_per_epoch' in config:
                    kwargs['steps_per_epoch'] = config['steps_per_epoch']
                if 'epochs' in config:
                    kwargs['epochs'] = config['epochs']
                # Optional parameters
                if 'pct_start' in config:
                    kwargs['pct_start'] = config['pct_start']
                if 'div_factor' in config:
                    kwargs['div_factor'] = config['div_factor']
                if 'final_div_factor' in config:
                    kwargs['final_div_factor'] = config['final_div_factor']
                    
            elif 'constant' in lr_scheduler_name:
                # ConstantLR: keeps LR constant (or with optional warmup factor)
                kwargs['factor'] = config.get('factor', 1.0)
                kwargs['total_iters'] = config.get('total_iters', 1)
                
            elif 'plateau' in lr_scheduler_name:
                # ReduceLROnPlateau needs factor, patience
                kwargs['factor'] = config.get('factor', 0.1)
                kwargs['patience'] = config.get('patience', 10)
                if 'mode' in config:
                    kwargs['mode'] = config['mode']
                    
            return cls(lr=lr, monitor_metric=monitor_metric, opt_type=opt_type, 
                      weight_decay=weight_decay, lr_scheduler_name=lr_scheduler_name,
                      warmup_ratio=warmup_ratio, warmup_epochs=warmup_epochs, **kwargs)
        return cls(lr=lr, monitor_metric=monitor_metric, opt_type=opt_type, weight_decay=weight_decay,
                  warmup_ratio=warmup_ratio, warmup_epochs=warmup_epochs)

    def __call__(self, model):
        optimizer = self.opt_fns[self.opt_type](model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        
        if self.lr_scheduler_name is None:
            return optimizer
        
        if self.lr_scheduler_name not in self.lr_schedulers:
            raise ValueError(f"Unknown scheduler: {self.lr_scheduler_name}")
        
        # Determine if warmup is needed (skip for OneCycleLR which has built-in warmup)
        use_warmup = (self.warmup_ratio > 0 or self.warmup_epochs is not None) and 'onecycle' not in self.lr_scheduler_name
        
        if use_warmup:
            # Calculate warmup epochs
            if self.warmup_epochs is not None:
                warmup_epochs = self.warmup_epochs
            else:
                # Get total epochs from kwargs (T_max for cosine, etc.)
                total_epochs = self.kwargs.get('T_max', self.kwargs.get('epochs', 100))
                warmup_epochs = max(1, int(total_epochs * self.warmup_ratio))
            
            # Create warmup scheduler (LinearLR: starts at lr*start_factor, linearly increases to lr)
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, 
                start_factor=0.1,  # Start at 10% of target LR
                total_iters=warmup_epochs
            )
            
            # Create main scheduler
            main_scheduler = self.lr_schedulers[self.lr_scheduler_name](optimizer, **self.kwargs)
            
            # Combine with SequentialLR
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, main_scheduler],
                milestones=[warmup_epochs]
            )
            print(f"[Warmup] Using {warmup_epochs} warmup epochs before {self.lr_scheduler_name}")
        else:
            # No warmup, use scheduler directly
            scheduler = self.lr_schedulers[self.lr_scheduler_name](optimizer, **self.kwargs)
        
        # Lightning scheduler config format
        scheduler_config = {
            "scheduler": scheduler,
            "monitor": f"val_{self.monitor_metric}",
        }
        
        # For ReduceLROnPlateau, add special flags to ensure proper timing
        if 'plateau' in self.lr_scheduler_name:
            # These settings ensure Lightning waits for validation before stepping
            scheduler_config["reduce_on_plateau"] = True
            scheduler_config["strict"] = False  # Don't crash if metric missing
        elif 'onecycle' in self.lr_scheduler_name:
            # OneCycleLR updates every step (batch), not epoch
            scheduler_config["interval"] = "step"
            scheduler_config["frequency"] = 1
        else:
            # For other schedulers, use standard epoch-based stepping
            scheduler_config["interval"] = "epoch"
            scheduler_config["frequency"] = 1
        
        return {
            "optimizer": optimizer, 
            "lr_scheduler": scheduler_config
        }

