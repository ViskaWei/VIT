import math
import torch
import torch.nn as nn
from transformers import ViTPreTrainedModel, ViTModel, ViTConfig
from transformers.modeling_outputs import SequenceClassifierOutput
from src.basemodule import BaseModel


def complete_with_orthogonal(U: torch.Tensor, out_dim: int) -> torch.Tensor:
    """
    给定 U: (in_dim, r)，补齐到 (in_dim, out_dim) 的正交基。
    若 r >= out_dim，直接截断返回前 out_dim 列。
    """
    in_dim, r = U.shape
    if r >= out_dim:
        return U[:, :out_dim]

    # 随机矩阵 + 拼接，然后做 QR 分解得到正交基
    rand_mat = torch.randn(in_dim, out_dim - r, device=U.device, dtype=U.dtype)
    A = torch.cat([U, rand_mat], dim=1)  # (in_dim, out_dim)
    Q, _ = torch.linalg.qr(A)  # (in_dim, out_dim)
    return Q[:, :out_dim]

def get_model(config):
    """Build ViT or ViT with a PCA-initialized global-attention front-end."""
    warmup_cfg = get_pca_config(config)
    vit_config = get_vit_config(config)
    loss_name = config.get('loss', {}).get('name', None)

    if warmup_cfg.get('global', False):
        r = warmup_cfg.get('r', None)   # Pass optional rank r from config for PCA init
        pca_path = warmup_cfg.get('global_pca_path', None)
        use_lora = bool(warmup_cfg.get('lora', False))
        # If r == 0, explicitly skip any PCA warm logic and do not load.
        if (r is not None) and int(r) == 0:
            pca_stats = None
        elif pca_path is not None and (r is None or int(r) > 0):
            # Support either a raw tensor (U) or a dict with more stats
            loaded = torch.load(pca_path, weights_only=True)
            if isinstance(loaded, dict):
                pca_stats = loaded
            else:
                pca_stats = {"U": loaded}
            # Optionally freeze Q/K for the first N epochs (train only V + downstream)
        else:
            pca_stats = None
        # Read freeze setting, but if PCA is not used, force-disable freeze.
        qk_freeze_epochs = int(warmup_cfg.get('freeze_qk_epochs', 0) or 0)
        if pca_stats is None:
            qk_freeze_epochs = 0
        
        return GlobalAttnViT(
            vit_config,
            pca_stats=pca_stats,
            loss_name=loss_name,
            r=r,
            use_lora=use_lora,
            qk_freeze_epochs=qk_freeze_epochs,
        )

    return MyViT(vit_config, loss_name=loss_name)

def get_vit_pretrain_model(config):
    vit_config = get_vit_config(config)
    # Decide loss name from config or sensible default by task
    loss_name = config.get('loss', {}).get('name', None)    
    vit_model = MyViT(vit_config, loss_name=loss_name)
    return vit_model

def get_pca_config(config):
    return config.get('warmup', {})


def get_vit_config(config):
    """Create a ViTConfig object from the provided config dict.

    For regression, if `data.param` is a comma-separated string or a list of parameter
    names (e.g., ['T_eff', 'log_g']), dynamically set `num_labels` to its length.
    """
    m = config['model']
    d = config.get('data', {})
    num_labels = int(m.get('num_labels', 1) or 1)
    task = (m.get('task_type') or m.get('task') or 'cls').lower()
    if task in ('reg', 'regression'):
        p = d.get('param', None)
        if isinstance(p, str) and len(p) > 0:
            plist = [x.strip() for x in p.split(',') if x.strip()]
            if len(plist) >= 1:
                num_labels = len(plist)
        elif isinstance(p, (list, tuple)) and len(p) > 0:
            num_labels = len(p)
        # Reflect back so downstream (e.g., metrics) can see it if they read config
        try:
            m['num_labels'] = num_labels
        except Exception:
            pass

    return ViTConfig(
        task_type=m['task_type'],
        image_size=m['image_size'],
        patch_size=m['patch_size'],
        num_channels=1,
        hidden_size=m['hidden_size'],
        num_hidden_layers=m['num_hidden_layers'],
        num_attention_heads=m['num_attention_heads'],
        intermediate_size=4 * m['hidden_size'],
        stride_ratio=m.get('stride_ratio', 1),
        # Optional explicit stride size; if provided, embedding layers will prefer it
        stride_size=m.get('stride_size', None),
        proj_fn=m['proj_fn'],
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        is_encoder_decoder=False,
        use_mask_token=False,
        qkv_bias=True,
        num_labels=num_labels,
    )


class MyViT(ViTPreTrainedModel, BaseModel):
    config_class = ViTConfig
    
    def __init__(self, config, loss_name="", model_name="ViT"):
        # Initialize HF parent (this calls nn.Module.__init__ once)
        ViTPreTrainedModel.__init__(self, config)
        self.config = config
        self.vit = ViTModel(config)
        self.vit.embeddings = MyEmbeddings(config)
        self.task_type = config.task_type
        if self.task_type == 'cls':  # classification
            self.classifier = nn.Linear(config.hidden_size, config.num_labels)
            self.loss_fct = nn.CrossEntropyLoss()
            loss_name = 'ce'
        elif self.task_type == 'reg':  # regression
            # Output dimension equals number of regression targets
            self.regressor = nn.Linear(config.hidden_size, config.num_labels)
            loss_name = loss_name or 'l2'
            self.loss_fct = nn.L1Loss() if ('l1' in (loss_name or '').lower()) else nn.MSELoss()

        # Initialize weights after modules are defined
        self.init_weights()
        # Compose a clear default model name (do NOT call BaseModel.__init__ here,
        # since it would invoke nn.Module.__init__ again and clear parameters.)
        # Prefer explicit stride_size for naming if provided; otherwise use stride_ratio
        _stride_used = getattr(config, 'stride_size', None)
        stride_tag = int(_stride_used) if (_stride_used is not None and _stride_used) else config.stride_ratio
        full_model_name = (
            f'{model_name}_p{config.patch_size}_h{config.hidden_size}_l{config.num_hidden_layers}_'
            f'a{config.num_attention_heads}_s{stride_tag}_p{config.proj_fn}'
        )
        self._model_name = full_model_name
        self._loss_name = loss_name or 'train'
        print(f'Creating {self._model_name} model with {self._loss_name} loss')

    def forward(self, pixel_values, labels=None, output_attentions=None, output_hidden_states=None, return_dict=None):
        return_dict = return_dict or self.config.use_return_dict
        
        outputs = self.vit(
            pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        # Use [CLS] token for prediction
        sequence_output = outputs[0]
        cls_token = sequence_output[:, 0, :]
        
        if self.task_type == 'cls':
            logits = self.classifier(cls_token)
        elif self.task_type == 'reg':
            logits = self.regressor(cls_token)
        
        loss = None
        if labels is not None:
            if self.task_type == 'cls':
                loss = self.loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))
            elif self.task_type == 'reg':
                loss = self.loss_fct(logits.view(-1), labels.view(-1).float())
        
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    # BaseModel interface stubs (not heavily used by current training loop)
    def compute_loss(self, *args, **kwargs):
        # Forward already returns loss when labels are provided
        out = self.forward(*args, **kwargs)
        return out.loss

    def log_outputs(self, outputs, log_fn=print, stage=''):
        # Log using BaseModel convention in case it's called
        if isinstance(outputs, dict) and 'loss' in outputs:
            log_fn({f'{self.loss_name}_loss': outputs['loss']})
        elif hasattr(outputs, 'loss'):
            log_fn({f'{self.loss_name}_loss': outputs.loss})


class MyWindowPatchEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.image_size = config.image_size # spectra length 
        self.patch_size = config.patch_size 
        self.num_channels = 1
        # If explicit stride_size is provided and > 0, prefer it; else derive from stride_ratio
        _stride_sz = getattr(config, 'stride_size', None)
        self.stride_size = int(_stride_sz) if (_stride_sz is not None and int(_stride_sz) > 0) else int(config.stride_ratio * self.patch_size)
        self.num_patches = math.ceil((self.image_size - self.patch_size) / self.stride_size) + 1
        self.projection = nn.Linear(self.patch_size, config.hidden_size)

    def forward(self, x):  # x size: (batch_size, spectra_length)
        batch_size = x.size(0)
        x = x.unfold(dimension=1, size=self.patch_size, step=self.stride_size)
        
        # 如果最后一个patch不完整，用0填充
        if x.size(1) < self.num_patches:
            padding = torch.zeros(batch_size, self.num_patches - x.size(1), self.patch_size, device=x.device)
            x = torch.cat([x, padding], dim=1)
        # Ensure contiguous memory before reshape/linear to avoid MPS buffer errors
        x = x.contiguous().reshape(batch_size, self.num_patches, self.patch_size)
        x = self.projection(x)
        return x  # (batch_size, num_patches, hidden_size)



class MyCNN1DPatchEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.image_size = config.image_size # spectra length = 2000
        self.patch_size = config.patch_size # patch_size = 100
        self.num_channels = 1
        _stride_sz = getattr(config, 'stride_size', None)
        self.stride_size = int(_stride_sz) if (_stride_sz is not None and int(_stride_sz) > 0) else int(config.stride_ratio * self.patch_size)
        self.num_patches = ((self.image_size - self.patch_size) // self.stride_size) + 1
        self.projection = nn.Conv1d(self.num_channels, config.hidden_size, kernel_size=self.patch_size, stride=self.stride_size)
    def forward(self, x):  
        x = x.reshape(-1, 1, self.image_size) # x size: (batch_size, spectra_length)
        x = self.projection(x)
        return x.transpose(1, 2)  # (batch_size, num_patches, hidden_size)


class MyEmbeddings(nn.Module):
    """Patch embeddings + [CLS] + learnable positional embeddings for 1D inputs."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        if self.config.proj_fn == 'SW':
            self.patch_embeddings = MyWindowPatchEmbeddings(config)
        elif self.config.proj_fn == 'C1D':
            self.patch_embeddings = MyCNN1DPatchEmbeddings(config)

        self.num_patches = self.patch_embeddings.num_patches
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.hidden_size))
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.position_embeddings = nn.Parameter(
            torch.randn(1, self.patch_embeddings.num_patches + 1, config.hidden_size)
        )

    def forward(self, x, bool_masked_pos=None, interpolate_pos_encoding=False):
        x = self.patch_embeddings(x)
        batch_size, _, _ = x.size()
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.position_embeddings
        x = self.dropout(x)
        return x

class GlobalAttentionLayer(nn.Module):
    def __init__(self, input_dim, pca_stats, r: int | None = None, use_lora: bool = False):
        super(GlobalAttentionLayer, self).__init__()
        self.input_dim = input_dim  # e.g., 4096
        self.use_lora = bool(use_lora)
        self.r = int(r) if (r is not None) else None
        if self.use_lora:
            if self.r is None or self.r <= 0:
                self.r = min(64, self.input_dim)
            rank = min(self.r, self.input_dim)
            self.rank = rank
            # Low-rank factorization: D->r->D
            self.q_down = nn.Linear(input_dim, rank, bias=False)
            self.q_up = nn.Linear(rank, input_dim, bias=False)
            self.k_down = nn.Linear(input_dim, rank, bias=False)
            self.k_up = nn.Linear(rank, input_dim, bias=False)
            self.v_down = nn.Linear(input_dim, rank, bias=False)
            self.v_up = nn.Linear(rank, input_dim, bias=False)
        else:
            # Full D->D projections like the original
            self.q_lin = nn.Linear(input_dim, input_dim, bias=False)
            self.k_lin = nn.Linear(input_dim, input_dim, bias=False)
            self.v_lin = nn.Linear(input_dim, input_dim, bias=False)
        self.softmax = nn.Softmax(dim=-1)

        # PCA initialization for Q/K
        if pca_stats is not None and ("U" in pca_stats):
            U = pca_stats["U"]  # U may be (r, D) or (D, r)
            if U.dim() != 2:
                raise ValueError("PCA U must be a 2D tensor")
            if U.size(0) == self.input_dim and U.size(1) <= self.input_dim:
                U_mat = U  # (D, r_like)
                inferred_r = U.size(1)
            elif U.size(1) == self.input_dim and U.size(0) <= self.input_dim:
                U_mat = U.t()  # transpose to (D, r_like)
                inferred_r = U.size(0)
            else:
                raise ValueError(f"Unexpected PCA U shape {tuple(U.shape)} for input_dim={self.input_dim}")

            if self.use_lora:
                use_r = min(self.rank, inferred_r)
                U_mat = U_mat[:, :use_r]  # (D, use_r)
                # Set down = U^T, up = U so that W ≈ U @ U^T (orthogonal projector)
                self.q_down.weight.data.zero_()
                self.q_up.weight.data.zero_()
                self.k_down.weight.data.zero_()
                self.k_up.weight.data.zero_()
                self.q_down.weight.data[:use_r, :].copy_(U_mat.t())
                self.q_up.weight.data[:, :use_r].copy_(U_mat)
                self.k_down.weight.data[:use_r, :].copy_(U_mat.t())
                self.k_up.weight.data[:, :use_r].copy_(U_mat)
                # Initialize V low-rank orthogonally
                nn.init.orthogonal_(self.v_down.weight)
                nn.init.orthogonal_(self.v_up.weight)
            else:
                # Full matrix: place PCA vectors in top rows of Q/K
                nn.init.orthogonal_(self.q_lin.weight)
                nn.init.orthogonal_(self.k_lin.weight)
                keep_r = min(inferred_r, self.r or inferred_r)
                self.q_lin.weight.data[:keep_r, :].copy_(U_mat.t()[:keep_r, :])
                self.k_lin.weight.data[:keep_r, :].copy_(U_mat.t()[:keep_r, :])
                nn.init.orthogonal_(self.v_lin.weight)

            # Store explained variance up to r if available
            self.explained_variance_at_r = None
            try:
                if isinstance(pca_stats, dict) and ("explained_variance_ratio" in pca_stats):
                    evr = pca_stats["explained_variance_ratio"]
                    use_val = (self.rank if self.use_lora else (self.r or evr.shape[0]))
                    keep_r = min(int(use_val), int(evr.shape[0]))
                    self.explained_variance_at_r = float(evr[:keep_r].sum().item())
                elif isinstance(pca_stats, dict) and ("S" in pca_stats):
                    S = pca_stats["S"]
                    total_var = float((S ** 2).sum().item())
                    use_val = (self.rank if self.use_lora else (self.r or S.shape[0]))
                    keep_r = min(int(use_val), int(S.shape[0]))
                    num = float((S[:keep_r] ** 2).sum().item())
                    self.explained_variance_at_r = num / total_var if total_var > 0 else None
            except Exception:
                self.explained_variance_at_r = None
        else:
            # Random init depending on mode
            if self.use_lora:
                nn.init.orthogonal_(self.q_down.weight)
                nn.init.orthogonal_(self.q_up.weight)
                nn.init.orthogonal_(self.k_down.weight)
                nn.init.orthogonal_(self.k_up.weight)
                nn.init.orthogonal_(self.v_down.weight)
                nn.init.orthogonal_(self.v_up.weight)
            else:
                std = 0.02
                nn.init.trunc_normal_(self.q_lin.weight, std=std)
                nn.init.trunc_normal_(self.k_lin.weight, std=std)
                nn.init.trunc_normal_(self.v_lin.weight, std=std)
            self.explained_variance_at_r = None

    def q_proj(self, x):
        if self.use_lora:
            return self.q_up(self.q_down(x))
        return self.q_lin(x)

    def k_proj(self, x):
        if self.use_lora:
            return self.k_up(self.k_down(x))
        return self.k_lin(x)

    def v_proj(self, x):
        if self.use_lora:
            return self.v_up(self.v_down(x))
        return self.v_lin(x)

    def forward(self, x):
        """
        Supports:
        - 2D input `(B, D)`: apply low-rank preconditioning via Q only
        - 3D input `(B, L, D)`: apply simple global attention over `L`
        """
        if x.dim() == 2:
            # (B, D) -> (B, D), pre-project with low-rank Q mapping
            return self.q_proj(x)

        # Expect (B, L, D) for attention
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.input_dim ** 0.5)
        attn_probs = self.softmax(attn_scores)
        attn_output = torch.matmul(attn_probs, v)
        return attn_output

    # --- Utilities for freezing/unfreezing Q/K during training ---
    def set_qk_trainable(self, trainable: bool = True):
        if self.use_lora:
            modules = (self.q_down, self.q_up, self.k_down, self.k_up)
        else:
            modules = (self.q_lin, self.k_lin)
        for m in modules:
            for p in m.parameters():
                p.requires_grad = trainable


class GlobalAttnViT(MyViT):
    """MyViT with a global PCA-initialized attention preconditioning layer.

    This subclass keeps HuggingFace/ViT behaviors (config, save_pretrained, etc.)
    and simply applies a linear/global-attention transform to the 1D input before
    delegating to the parent `MyViT` forward.
    """

    def __init__(self, config, pca_stats=None, loss_name=None, model_name="GAtt_ViT", r: int | None = None, use_lora: bool = False, qk_freeze_epochs: int = 0):
        tag = (f"Lo{r}" if use_lora else f"Fc{r}") if pca_stats is not None else ("LoRD" if use_lora else "FcRD")
        model_name = f"Gpca{tag}_fz{qk_freeze_epochs}_{model_name}"
        super().__init__(config, loss_name=loss_name, model_name=model_name)
        self.attn = GlobalAttentionLayer(input_dim=config.image_size, pca_stats=pca_stats, r=r, use_lora=use_lora)
        # Freeze Q/K for the first N epochs if requested
        self.qk_freeze_epochs = int(qk_freeze_epochs or 0)
        self._qk_frozen_state = None  # track last applied state

    def apply_qk_freeze(self, current_epoch: int) -> bool:
        """Freeze Q/K for the first `qk_freeze_epochs` epochs.
        Returns True if Q/K are frozen for this epoch, else False.
        """
        # If no freeze is requested, avoid touching any state.
        if self.qk_freeze_epochs <= 0:
            return False
        should_freeze = (current_epoch < self.qk_freeze_epochs)
        if should_freeze != self._qk_frozen_state:
            # Transition state only when it changes
            self.attn.set_qk_trainable(not should_freeze)
            self._qk_frozen_state = should_freeze
        return should_freeze

    def forward(self,
                pixel_values,
                labels=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None):
        # Apply global attention/preconditioning on the raw 1D signal
        x = self.attn(pixel_values)
        # Then call standard MyViT forward
        return super().forward(
            x,
            labels=labels,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
