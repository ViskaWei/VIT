from __future__ import annotations

import torch.nn as nn
from transformers import ViTConfig, ViTModel, ViTPreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput

from src.basemodule import BaseModel

from .embedding import SpectraEmbeddings
from .model_utils import build_model_name


__all__ = ["MyViT"]


class MyViT(ViTPreTrainedModel, BaseModel):
    """Vision Transformer with optional input preprocessor"""
    
    config_class = ViTConfig

    def __init__(
        self,
        config: ViTConfig,
        loss_name: str = "",
        model_name: str = "ViT",
        preprocessor: nn.Module | None = None,
    ) -> None:
        ViTPreTrainedModel.__init__(self, config)
        self.config = config
        self.vit = ViTModel(config)
        self.vit.embeddings = SpectraEmbeddings(config)
        self.preprocessor = preprocessor
        
        # Setup task-specific head and loss
        self.task_type = config.task_type
        if self.task_type == "cls":
            self.classifier = nn.Linear(config.hidden_size, config.num_labels)
            self.loss_fct = nn.CrossEntropyLoss()
            loss_name = "ce"
        elif self.task_type == "reg":
            self.regressor = nn.Linear(config.hidden_size, config.num_labels)
            loss_name = loss_name or "l2"
            self.loss_fct = nn.L1Loss() if "l1" in loss_name.lower() else nn.MSELoss()
        else:
            raise ValueError(f"Unsupported task_type '{self.task_type}'")

        self.init_weights()
        
        self._model_name = build_model_name(config, model_name)
        self._loss_name = loss_name

    def forward(self, pixel_values, labels=None, output_attentions=None, output_hidden_states=None, return_dict=None):
        return_dict = return_dict or self.config.use_return_dict

        # Apply preprocessor if exists
        if self.preprocessor is not None:
            pixel_values = self.preprocessor(pixel_values)

        # ViT forward pass
        outputs = self.vit(pixel_values, output_attentions=output_attentions, 
                          output_hidden_states=output_hidden_states, return_dict=return_dict)
        cls_token = outputs[0][:, 0, :]

        # Task-specific head
        logits = self.classifier(cls_token) if self.task_type == "cls" else self.regressor(cls_token)

        # Compute loss
        loss = None
        if labels is not None:
            if self.task_type == "cls":
                loss = self.loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))
            else:  # reg
                loss = self.loss_fct(logits.view(-1), labels.view(-1).float())

        return SequenceClassifierOutput(
            loss=loss, logits=logits, 
            hidden_states=outputs.hidden_states, attentions=outputs.attentions
        )

    def compute_loss(self, *args, **kwargs):
        return self.forward(*args, **kwargs).loss

    def log_outputs(self, outputs, log_fn=print, stage: str = ""):
        loss = outputs.get("loss") if isinstance(outputs, dict) else getattr(outputs, "loss", None)
        if loss is not None:
            log_fn({f"{self.loss_name}_loss": loss})

    def set_preprocessor_trainable(self, trainable: bool) -> None:
        """Freeze/unfreeze preprocessor parameters"""
        if self.preprocessor is None:
            return
        
        if hasattr(self.preprocessor, "set_qk_trainable"):
            self.preprocessor.set_qk_trainable(trainable)
        elif hasattr(self.preprocessor, "freeze"):
            self.preprocessor.freeze(not trainable)
        else:
            for param in self.preprocessor.parameters():
                param.requires_grad = trainable
