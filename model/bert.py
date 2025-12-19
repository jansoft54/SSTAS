import torch
import numpy as np

import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from dataclasses import dataclass
from model.attention import RotaryEmbedding, rotate_half, apply_rotary_pos_emb, RotaryMHA


@dataclass
class ActionBERTConfig:
    total_classes: int = 10
    input_dim: int = 2048
    d_model: int = 128
    num_heads: int = 8
    num_layers: int = 4
    ffn_dim: int = 128
    dropout: float = 0.05

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
        self.norm = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(0.2)
        self.rotary_mha = RotaryMHA(config=config)
        
    
    def create_windows(self,x,padding_mask,cos,sin):
        B, T, C = x.shape
        W = self.local_window_size
        
        cos_t = cos.squeeze(0).squeeze(0) 
        sin_t = sin.squeeze(0).squeeze(0)
        cos_t = cos_t.unsqueeze(0).expand(B, -1, -1)
        sin_t = sin_t.unsqueeze(0).expand(B, -1, -1)
        
        pad_end = (W - (T % W)) % W
        if pad_end > 0:
            x_padded = F.pad(x, (0, 0, 0, pad_end))
            padding_mask_padded = F.pad(padding_mask, (0, pad_end), value=False)
            
            cos_padded = F.pad(cos_t, (0, 0, 0, pad_end)) 
            sin_padded = F.pad(sin_t, (0, 0, 0, pad_end))
        else:
            x_padded = x
            padding_mask_padded = padding_mask
            cos_padded = cos_t
            sin_padded = sin_t
        
        q_windows = x_padded.unfold(1, W, W).permute(0, 1, 3, 2).reshape(-1, W, C)

        cos_q = cos_padded.unfold(1, W, W).permute(0, 1, 3, 2).reshape(-1, W, cos_padded.shape[-1]).unsqueeze(1)
        sin_q = sin_padded.unfold(1, W, W).permute(0, 1, 3, 2).reshape(-1, W, sin_padded.shape[-1]).unsqueeze(1)
        
        
        half_pad = W // 2
        x_context = F.pad(x_padded, (0, 0, half_pad, half_pad))
        cos_context = F.pad(cos_padded, (0, 0, half_pad, half_pad))
        sin_context = F.pad(sin_padded, (0, 0, half_pad, half_pad))
        
        
                            
        k_windows = x_context.unfold(1, 2*W, W).permute(0, 1, 3, 2).reshape(-1, 2*W, C)
        padding_mask_k = F.pad(padding_mask_padded, (half_pad, half_pad), value=False)
        padding_mask_k = padding_mask_k.unfold(1, 2*W, W).reshape(-1, 2*W)
        cos_k = cos_context.unfold(1, 2*W, W).permute(0, 1, 3, 2).reshape(-1, 2*W, cos_padded.shape[-1]).unsqueeze(1)
        sin_k = sin_context.unfold(1, 2*W, W).permute(0, 1, 3, 2).reshape(-1, 2*W, sin_padded.shape[-1]).unsqueeze(1)

    
        q = q_windows
        k = k_windows
        pk = padding_mask_k
        
        return q, k, pk, cos_q, sin_q, cos_k, sin_k
    
    def forward(self, x,padding_mask,cos=None,sin=None):
        batch_size, seq_len, dim = x.size()
        q, k, pk ,cos_q, sin_q, cos_k, sin_k= self.create_windows(x,padding_mask=padding_mask,cos=cos,sin=sin)
        v = k

        attn_mask = pk
        rows_with_no_data = ~attn_mask.any(dim=-1)
        
        if rows_with_no_data.any():
            attn_mask[rows_with_no_data, 0] = True 
   
        attn_output = self.rotary_mha(q, k, v, cos_q, sin_q, cos_k, sin_k, key_padding_mask=attn_mask)
        attn_output = attn_output.view(batch_size, -1, self.local_window_size, dim)
        
        attn_output = attn_output.flatten(1, 2)[:,:seq_len,:]
        if torch.isnan(attn_output).any():
            raise Exception("NaN values detected in LOCAL attention output")
     
        return attn_output.contiguous()


class GlobalAttention(nn.Module):
    def __init__(self,config: ActionBERTConfig,window_dilation=8):
        super().__init__()
        self.rotary_mha = RotaryMHA(config=config)
        self.norm = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(0.2)
        self.window_dilation = window_dilation
        
        
    def create_windows(self,x,padding_mask,cos,sin):
        B,T,C = x.shape
        W = self.window_dilation
        pad_end = (W - (T % W)) % W
        
        cos_t = cos.squeeze(0).squeeze(0).unsqueeze(0).expand(B, -1, -1) # [B, T, HeadDim]
        sin_t = sin.squeeze(0).squeeze(0).unsqueeze(0).expand(B, -1, -1)
      
        if pad_end > 0:
            x_padded = F.pad(x, (0, 0, 0, pad_end))
            padding_mask_padded = F.pad(padding_mask, (0, pad_end), value=False)
            cos_padded = F.pad(cos_t, (0, 0, 0, pad_end))
            sin_padded = F.pad(sin_t, (0, 0, 0, pad_end))
        else:
            x_padded = x
            padding_mask_padded = padding_mask
            cos_padded = cos_t
            sin_padded = sin_t
            
        windows = x_padded.unfold(1, W, W).permute(0, 3, 1, 2).contiguous()
        windows = windows.flatten(0,1)
        
        
        padding_mask_padded = padding_mask_padded.unfold(1, W, W).transpose(1,2).contiguous()
        padding_mask_padded = padding_mask_padded.flatten(0,1)
       
        cos_windows = cos_padded.unfold(1, W, W).permute(0, 3, 1, 2).contiguous()
        cos_windows = cos_windows.flatten(0, 1).unsqueeze(1)
        
        sin_windows = sin_padded.unfold(1, W, W).permute(0, 3, 1, 2).contiguous()
        sin_windows = sin_windows.flatten(0, 1).unsqueeze(1)

        q,k,v = windows, windows,windows
        pk = padding_mask_padded
        return q, k, v, pk, cos_windows, sin_windows
    
    def forward(self, x,padding_mask,cos=None,sin=None):
        B,T,C = x.shape
        
        q, k, v, pk, cos_windows, sin_windows = self.create_windows(x,padding_mask=padding_mask,cos=cos,sin=sin)
    
        attn_mask = pk
        rows_with_no_data = ~attn_mask.any(dim=-1)
        if rows_with_no_data.any():
            attn_mask[rows_with_no_data, 0] = True 
        attn_output = self.rotary_mha(x_q=q, 
            x_k=k, 
            x_v=v, 
            cos_q=cos_windows, 
            sin_q=sin_windows, 
            cos_k=cos_windows, 
            sin_k=sin_windows, 
            key_padding_mask=attn_mask)
        
        attn_output = attn_output.view(B, self.window_dilation,-1, C)
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous().view(B, -1, C)
        attn_output = attn_output[:,:T,:]
        if torch.isnan(attn_output).any():
            raise Exception("NaN values detected in GLOBAL attention output")
        return attn_output.contiguous()
    
class Block(nn.Module):
    def __init__(self,config: ActionBERTConfig):
        super().__init__()
        self.local_attn = LocalAttention(config=config,local_window_size=128)
        self.global_attn = GlobalAttention(config=config,window_dilation=64)
        
        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)
        self.norm3 = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(0.2)
        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, config.d_model * 4),
            nn.GELU(),
            nn.Linear(config.d_model * 4, config.d_model),
        )
    def forward(self, x,padding_mask,cos=None,sin=None):
        """
        x: [B, T, D]
        """
        resid = x
        x_norm = self.norm1(x)
        local_out = self.local_attn(x_norm, padding_mask, cos, sin)
        x = resid + self.dropout(local_out)
        
        
        resid = x
        x_norm = self.norm2(x)
        global_out = self.global_attn(x_norm, padding_mask, cos, sin)
        x = resid + self.dropout(global_out)

  
        resid = x
        x_norm = self.norm3(x)
        ffn_out = self.ffn(x_norm)
        x = resid + self.dropout(ffn_out)
        return x

class ActionBERT(nn.Module):
    def __init__(self, 
                 config: ActionBERTConfig,   
               ):
        super().__init__()
        self.config = config
        self.input_proj = nn.Linear(config.input_dim,config.d_model)
        self.output_head = nn.Linear(config.d_model, config.input_dim)
        self.prototypes = nn.Linear(config.d_model, config.total_classes,bias=False)
        self.final_norm = nn.LayerNorm(config.d_model)

        self.boundary_head = nn.Sequential(
            nn.LayerNorm(config.d_model),
            nn.Linear(config.d_model, config.d_model*2),
            nn.GELU(),
            nn.Linear(config.d_model*2, config.d_model//2),
            nn.GELU(),
            nn.Linear(config.d_model//2, 2)
        )
        
        self.rotary_emb = RotaryEmbedding(config.d_model // config.num_heads)
        self.masking_module = MaskingModule(config.input_dim,config.d_model)

        self.layers = nn.ModuleList([
            Block(config)
            for _ in range(config.num_layers)
        ])

    def forward(self, x,patch_mask,padding_mask):
        seq_len = x.shape[1]
        cos, sin = self.rotary_emb(x, seq_len=seq_len)
      #  cos = torch.zeros_like(cos)
       # sin = torch.zeros_like(sin)
        x = self.input_proj(x) 
        x = self.masking_module(x,mask=patch_mask)
        for layer in self.layers:
            x = layer(x,padding_mask=padding_mask,cos=cos,sin=sin)
        
        x = self.final_norm(x)

   
        recon_features = self.output_head(x)
        prototype_logits = self.prototypes(x)
        # = self.boundary_head(x)
        return recon_features, prototype_logits,   x 


