import torch
import numpy as np

import torch
import torch.nn as nn
import math
import torch.nn.functional as F


class ActionBERTConfig:
    def __init__(self,
                 total_classes= 10,
                 input_dim=2048,
                 d_model=128,
                 num_heads=8,
                 num_layers=4,
                 ffn_dim=128,
                 dropout=0.1):
        self.total_classes = total_classes
        self.input_dim = input_dim
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.ffn_dim = ffn_dim
        self.dropout = dropout

class MaskingModule(nn.Module):
    def __init__(self,input_dim,model_dim):
        super().__init__()
        self.input_dim = input_dim
        self.mask_value = model_dim
        self.mask_token = nn.Parameter(torch.randn(1, 1, model_dim) * 0.02)

        
    def forward(self,features,mask):
        features_masked = features.clone()
        token = self.mask_token.type_as(features_masked) 

        features_masked[mask] = token.squeeze()
        return features_masked
    
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=20000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)

        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x):
      
        T = x.size(1)
    
        return x + self.pe[:T].unsqueeze(0)  

class LocalAttention(nn.Module):
    def __init__(self,config: ActionBERTConfig, local_window_size=64):
        
        super().__init__()
        self.local_window_size = local_window_size
        self.mha = nn.MultiheadAttention(config.d_model, config.num_heads, dropout=0, batch_first=True)
        self.norm = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(0.1)
        
    
    def create_windows(self,x,padding_mask):
        _, T, C = x.shape
        W = self.local_window_size
        pad_end = (W - (T % W)) % W
        if pad_end > 0:
            x_padded = F.pad(x, (0, 0, 0, pad_end))
            padding_mask_padded = F.pad(padding_mask, (0, pad_end), value=True)
        else:
            x_padded = x
            padding_mask_padded = padding_mask
        
        q_windows = x_padded.unfold(1, W, W).permute(0, 1, 3, 2).reshape(-1, W, C)
        padding_mask_q = padding_mask_padded.unfold(1, W, W).reshape(-1, W)
        
        half_pad = W // 2
        x_context = F.pad(x_padded, (0, 0, half_pad, half_pad))
        k_windows = x_context.unfold(1, 2*W, W).permute(0, 1, 3, 2).reshape(-1, 2*W, C)
        padding_mask_k = F.pad(padding_mask_padded, (half_pad, half_pad), value=True)
        padding_mask_k = padding_mask_k.unfold(1, 2*W, W).reshape(-1, 2*W)
        
        q = q_windows
        k = k_windows
        v = k_windows
        pq = padding_mask_q
        pk = padding_mask_k
        pv = padding_mask_k
        
        return q, k, v, pq, pk, pv
    
    def forward(self, x,padding_mask):
        batch_size, seq_len, dim = x.size()
        q, k, v, _, pk, _ = self.create_windows(x,padding_mask=padding_mask)
        attn_mask = ~pk 
        all_masked_rows = attn_mask.all(dim=-1)
        
        if all_masked_rows.any():
            attn_mask[all_masked_rows, 0] = False 
        attn_output, _ = self.mha(q, k, v, key_padding_mask=attn_mask)
        
        attn_output = attn_output.view(batch_size, -1, self.local_window_size, dim)
        attn_output = attn_output.flatten(1, 2)[:,:seq_len,:]
        if torch.isnan(attn_output).any():
            raise Exception("NaN values detected in LOCAL attention output")
        return attn_output.contiguous()


class GlobalAttention(nn.Module):
    def __init__(self,config: ActionBERTConfig,window_dilation=8):
        super().__init__()
        self.mha = nn.MultiheadAttention(config.d_model, config.num_heads, dropout=0, batch_first=True)
        self.norm = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(0.1)
        self.window_dilation = window_dilation
        
        
    def create_windows(self,x,padding_mask):
        B,T,C = x.shape
        W = self.window_dilation
        pad_end = (W - (T % W)) % W
        if pad_end > 0:
            x_padded = F.pad(x, (0, 0, 0, pad_end))
            padding_mask_padded = F.pad(padding_mask, (0, pad_end), value=True)
        else:
            x_padded = x
            padding_mask_padded = padding_mask
        
        windows = x_padded.unfold(1, W, W).permute(0, 3, 1, 2).contiguous()
        windows = windows.flatten(0,1)
        padding_mask_padded = padding_mask_padded.unfold(1, W, W).transpose(1,2).contiguous()
        padding_mask_padded = padding_mask_padded.flatten(0,1)
       
        q,k,v = windows, windows,windows
        pq,pk,pv = padding_mask_padded,padding_mask_padded,padding_mask_padded
        return q, k, v, pq, pk, pv
    
    def forward(self, x,padding_mask):
        B,T,C = x.shape
        
        q, k, v, _, pk, _ = self.create_windows(x,padding_mask=padding_mask)

        attn_output, _ = self.mha(q, k, v, key_padding_mask=~pk)

        attn_output = attn_output.view(B, self.window_dilation,-1, C)
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous().view(B, -1, C)
        attn_output = attn_output[:,:T,:]
        if torch.isnan(attn_output).any():
            raise Exception("NaN values detected in GLOBAL attention output")
            attn_output = torch.nan_to_num(attn_output, nan=0.0)        
        return attn_output.contiguous()
    
class Block(nn.Module):
    def __init__(self,config: ActionBERTConfig):
        super().__init__()
        self.local_attn = LocalAttention(config=config,local_window_size=64)
        self.global_attn = GlobalAttention(config=config,window_dilation=16)
        
        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)
        self.norm3 = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(0.1)
        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, config.d_model * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(config.d_model * 4, config.d_model),
        )
    def forward(self, x,padding_mask):
        """
        x: [B, T, D]
        """
        attn_input = self.norm1(x)
        local_out = self.local_attn(attn_input, padding_mask)
        x = x + self.dropout(local_out) 
        global_input = self.norm2(x)
        global_out = self.global_attn(global_input, padding_mask)
        x = x + self.dropout(global_out)

      
        ffn_input = self.norm3(x)
        ffn_out = self.ffn(ffn_input)
        x = x + self.dropout(ffn_out)
        return x





class ActionBERT(nn.Module):
    def __init__(self, 
                 config: ActionBERTConfig,   
               ):
        super().__init__()
        self.config = config
        self.input_proj = nn.Linear(config.input_dim,config.d_model)
        self.output_head = nn.Linear(config.d_model, config.input_dim)
        self.prototypes = nn.Linear(config.d_model, config.total_classes)

        self.boundary_head = nn.Sequential(
            nn.LayerNorm(config.d_model),
            nn.Linear(config.d_model, config.d_model*2),
            nn.GELU(),
            nn.Linear(config.d_model*2, config.d_model//2),
            nn.GELU(),
            nn.Linear(config.d_model//2, 2)
        )
        
        self.pos_encoding = PositionalEncoding(config.d_model)
        self.masking_module = MaskingModule(config.input_dim,config.d_model)

        self.layers = nn.ModuleList([
            Block(config)
            for _ in range(config.num_layers)
        ])

    def forward(self, x,patch_mask,padding_mask):
        x = self.input_proj(x) 
        x = self.masking_module(x,mask=patch_mask)
        x = self.pos_encoding(x)
        for layer in self.layers:
            x = layer(x,padding_mask=padding_mask)
            
        boundaries = self.boundary_head(x)
        recon_features = self.output_head(x)
        prototype_logits = self.prototypes(x)
        return recon_features, prototype_logits, boundaries
