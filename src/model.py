import math
import torch
import torch.nn as nn
from transformers import ViTPreTrainedModel, ViTModel, ViTConfig
from transformers.modeling_outputs import SequenceClassifierOutput

class MyViT(ViTPreTrainedModel):
    config_class = ViTConfig
    
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.vit = ViTModel(config)
        self.vit.embeddings = MyEmbeddings(config)
        self.task_type = config.task_type
        if self.task_type == 'cls': # classification
            self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        elif self.task_type == 'reg':  # regression
            self.regressor = nn.Linear(config.hidden_size, 1)
        
        self.init_weights()
        self.loss_name = 'train'
        self.name = f'ViT_p{config.patch_size}_h{config.hidden_size}_l{config.num_hidden_layers}_a{config.num_attention_heads}_s{config.stride_ratio}_p{config.proj_fn}'

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
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))
            elif self.task_type == 'reg':  # regression
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1).float())
        
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class MyWindowPatchEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.image_size = config.image_size # spectra length 
        self.patch_size = config.patch_size 
        self.num_channels = 1
        self.stride_size = int(config.stride_ratio * self.patch_size)
        self.num_patches = math.ceil((self.image_size - self.patch_size) / self.stride_size) + 1
        self.projection = nn.Linear(self.patch_size, config.hidden_size)

    def forward(self, x):  # x size: (batch_size, spectra_length)
        batch_size = x.size(0)
        x = x.unfold(dimension=1, size=self.patch_size, step=self.stride_size)
        
        # 如果最后一个patch不完整，用0填充
        if x.size(1) < self.num_patches:
            padding = torch.zeros(batch_size, self.num_patches - x.size(1), self.patch_size, device=x.device)
            x = torch.cat([x, padding], dim=1)
        x = x.view(batch_size, self.num_patches, self.patch_size)
        x = self.projection(x)
        return x  # (batch_size, num_patches, hidden_size)



class MyCNN1DPatchEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.image_size = config.image_size # spectra length = 2000
        self.patch_size = config.patch_size # patch_size = 100
        self.num_channels = 1
        self.stride_size = int(config.stride_ratio * self.patch_size)
        self.num_patches = ((self.image_size - self.patch_size) // self.stride_size) + 1
        self.projection = nn.Conv1d(self.num_channels, config.hidden_size, kernel_size=self.patch_size, stride=self.stride_size)
    def forward(self, x):  
        x = x.reshape(-1, 1, self.image_size) # x size: (batch_size, spectra_length)
        x = self.projection(x)
        return x.transpose(1, 2)  # (batch_size, num_patches, hidden_size)


class MyEmbeddings(nn.Module):
    """
    Combine the patch embeddings with the class token and position embeddings.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        if self.config.proj_fn == 'SW':
            self.patch_embeddings = MyWindowPatchEmbeddings(config)
        elif self.config.proj_fn == 'C1D':
            self.patch_embeddings = MyCNN1DPatchEmbeddings(config)

        self.num_patches = self.patch_embeddings.num_patches
        # Create a learnable [CLS] token
        # Similar to BERT, the [CLS] token is added to the beginning of the input sequence
        # and is used to classify the entire sequence
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.hidden_size))
        # Create position embeddings for the [CLS] token and the patch embeddings
        # Add 1 to the sequence length for the [CLS] token
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # Embedding function map
        embed_fn_map = {
            "APE": self.ape_embedding,
            "CPE": self.cpe_embedding,
            "RPE": self.rpe_embedding,
            "RoPE": self.rope_embedding,
            "VPE": nn.Parameter(torch.randn(1, self.patch_embeddings.num_patches + 1, config.hidden_size))
        }
        self.position_embeddings = embed_fn_map['VPE']


    def ape_embedding(self, x):
        position = torch.arange(self.num_patches + 1, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.config.hidden_size, 2).float() * (-math.log(10000.0) / self.config.hidden_size))
        pe = torch.zeros(1, self.num_patches + 1, self.config.hidden_size)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        return pe.to(x.device)

    def cpe_embedding(self, x):
        # Implement CPE
        batch_size, seq_len, hidden_size = x.size()
        position = torch.arange(1, seq_len + 1, dtype=torch.float, device=x.device).unsqueeze(0).unsqueeze(-1)
        position = position.expand(batch_size, -1, hidden_size)
        return position

    def rpe_embedding(self, x):
        # Implement RPE
        return x

    def rope_embedding(self, x):
        # Implement RoPE
        return x

    def forward(self, x, bool_masked_pos=None, interpolate_pos_encoding=False):
        x = self.patch_embeddings(x)
        batch_size, _, _ = x.size()
        # Expand the [CLS] token to the batch size
        # (1, 1, hidden_size) -> (batch_size, 1, hidden_size)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        # Concatenate the [CLS] token to the beginning of the input sequence
        # This results in a sequence length of (num_patches + 1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.position_embeddings
        x = self.dropout(x)
        return x
