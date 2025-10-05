from __future__ import annotations

import torch
import torch.nn as nn
from transformers import ViTConfig, ViTModel, ViTPreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput

from src.basemodule import BaseModel

from .attention import GlobalAttentionLayer
from .embedding import SpectraEmbeddings


__all__ = ["MyViT", "PreconditionedViT", "GlobalAttnViT"]


class MyViT(ViTPreTrainedModel, BaseModel):
    config_class = ViTConfig

    def __init__(self, config, loss_name: str = "", model_name: str = "ViT") -> None:
        ViTPreTrainedModel.__init__(self, config)
        self.config = config
        self.vit = ViTModel(config)
        self.vit.embeddings = SpectraEmbeddings(config)
        self.embed_freeze_epochs: int = 0
        self._embed_frozen_state: bool | None = None
        self.task_type = config.task_type
        if self.task_type == "cls":
            self.classifier = nn.Linear(config.hidden_size, config.num_labels)
            self.loss_fct = nn.CrossEntropyLoss()
            loss_name = "ce"
        elif self.task_type == "reg":
            self.regressor = nn.Linear(config.hidden_size, config.num_labels)
            loss_name = loss_name or "l2"
            self.loss_fct = nn.L1Loss() if ("l1" in (loss_name or "").lower()) else nn.MSELoss()

        self.init_weights()
        stride_used = getattr(config, "stride_size", None)
        stride_tag = int(stride_used) if (stride_used is not None and stride_used) else config.stride_ratio
        full_model_name = (
            f"{model_name}_p{config.patch_size}_h{config.hidden_size}_l{config.num_hidden_layers}_"
            f"a{config.num_attention_heads}_s{stride_tag}_p{config.proj_fn}"
        )
        self._model_name = full_model_name
        self._loss_name = loss_name or "train"
        print(f"Creating {self._model_name} model with {self._loss_name} loss")

    def _apply_input_preprocessor(self, pixel_values: torch.Tensor) -> torch.Tensor:
        preprocessor = getattr(self, "preprocessor", None)
        if isinstance(preprocessor, nn.Module):
            return preprocessor(pixel_values)
        return pixel_values

    def forward(
        self,
        pixel_values,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict or self.config.use_return_dict

        pixel_values = self._apply_input_preprocessor(pixel_values)

        outputs = self.vit(
            pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        cls_token = sequence_output[:, 0, :]

        if self.task_type == "cls":
            logits = self.classifier(cls_token)
        elif self.task_type == "reg":
            logits = self.regressor(cls_token)
        else:
            raise ValueError(f"Unsupported task_type '{self.task_type}'")

        loss = None
        if labels is not None:
            if self.task_type == "cls":
                loss = self.loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))
            elif self.task_type == "reg":
                loss = self.loss_fct(logits.view(-1), labels.view(-1).float())

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def compute_loss(self, *args, **kwargs):
        out = self.forward(*args, **kwargs)
        return out.loss

    def log_outputs(self, outputs, log_fn=print, stage: str = ""):
        if isinstance(outputs, dict) and "loss" in outputs:
            log_fn({f"{self.loss_name}_loss": outputs["loss"]})
        elif hasattr(outputs, "loss"):
            log_fn({f"{self.loss_name}_loss": outputs.loss})

    def apply_embed_freeze(self, current_epoch: int) -> bool:
        if int(self.embed_freeze_epochs or 0) <= 0:
            return False
        should_freeze = current_epoch < int(self.embed_freeze_epochs)
        if should_freeze != self._embed_frozen_state:
            try:
                self.vit.embeddings.set_patch_proj_trainable(not should_freeze)
            except Exception:
                pass
            self._embed_frozen_state = should_freeze
        return should_freeze


class PreconditionedViT(MyViT):
    """ViT variant that applies an input preprocessor before the encoder."""

    def __init__(
        self,
        config,
        *,
        preprocessor: nn.Module,
        loss_name=None,
        model_name: str = "Precond_ViT",
        freeze_epochs: int = 0,
    ) -> None:
        self.preprocessor_freeze_epochs = int(freeze_epochs or 0)
        self._preprocessor_frozen_state: bool | None = None
        super().__init__(config, loss_name=loss_name, model_name=model_name)
        self.preprocessor = preprocessor
        # Maintain legacy attribute names for scheduler utilities.
        self.qk_freeze_epochs = int(self.preprocessor_freeze_epochs)
        self._qk_frozen_state: bool | None = None

    def _set_preprocessor_trainable(self, trainable: bool) -> None:
        module = getattr(self, "preprocessor", None)
        if not isinstance(module, nn.Module):
            return
        if hasattr(module, "set_qk_trainable"):
            module.set_qk_trainable(trainable)
        elif hasattr(module, "freeze"):
            module.freeze(not trainable)
        else:
            for param in module.parameters():
                param.requires_grad = trainable

    def apply_preprocessor_freeze(self, current_epoch: int) -> bool:
        if self.preprocessor_freeze_epochs <= 0:
            return False
        should_freeze = current_epoch < self.preprocessor_freeze_epochs
        if should_freeze != self._preprocessor_frozen_state:
            self._set_preprocessor_trainable(not should_freeze)
            self._preprocessor_frozen_state = should_freeze
        return should_freeze

    def apply_qk_freeze(self, current_epoch: int) -> bool:  # Backward compatibility
        return self.apply_preprocessor_freeze(current_epoch)


class GlobalAttnViT(PreconditionedViT):
    def __init__(
        self,
        config,
        pca_stats=None,
        loss_name=None,
        model_name: str = "GAtt_ViT",
        r: int | None = None,
        use_lora: bool = False,
        qk_freeze_epochs: int = 0,
        UV: str = "",
        use_input_bias: bool = False,
    ) -> None:
        tag = (f"Lo{r}" if use_lora else f"Fc{r}") if pca_stats is not None else ("LoRD" if use_lora else "FcRD")
        model_name = f"G{UV}V{tag}_fz{qk_freeze_epochs}_{model_name}"
        attn = GlobalAttentionLayer(
            input_dim=config.image_size,
            pca_stats=pca_stats,
            r=r,
            use_lora=use_lora,
            uv_key=UV if isinstance(UV, str) and len(UV) > 0 else None,
            use_pca_bias=bool(use_input_bias),
        )
        super().__init__(
            config,
            preprocessor=attn,
            loss_name=loss_name,
            model_name=model_name,
            freeze_epochs=qk_freeze_epochs,
        )
        self.attn = attn
        try:
            print(
                f"[global-warmup] Global attention preprocessor initialized: name='{self._model_name}', UV={UV}, r={r}, lora={use_lora}, bias={use_input_bias}, qk_freeze_epochs={self.qk_freeze_epochs}"
            )
        except Exception:
            pass
