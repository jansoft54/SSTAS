import torch
import numpy as np

import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=20000, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len_cached = max_position_embeddings
        
        t = torch.arange(max_position_embeddings, device=inv_freq.device).type_as(inv_freq)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, x, seq_len=None):
        if seq_len > self.max_seq_len_cached:
            self.max_seq_len_cached = seq_len
            t = torch.arange(seq_len, device=self.inv_freq.device).type_as(self.inv_freq)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            self.cos_cached = emb.cos()[None, None, :, :]
            self.sin_cached = emb.sin()[None, None, :, :]
            
        return self.cos_cached[:, :, :seq_len, ...], self.sin_cached[:, :, :seq_len, ...]

def rotate_half(x):
    """Hilfsfunktion: Rotiert die Hälfte der Dimensionen."""
    x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos_q, sin_q, cos_k, sin_k):
    """
    Wendet RoPE an.
    Akzeptiert unterschiedliche Positions-Embeddings für Query und Key.
    
    q: [Batch, Heads, T_Query, Dim]
    k: [Batch, Heads, T_Key, Dim]
    cos_q, sin_q: [Batch, 1, T_Query, Dim]
    cos_k, sin_k: [Batch, 1, T_Key, Dim]
    """
    q_embed = (q * cos_q) + (rotate_half(q) * sin_q)    
    k_embed = (k * cos_k) + (rotate_half(k) * sin_k)
    
    return q_embed, k_embed

class RotaryMHA(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_heads
        self.d_model = config.d_model
        self.head_dim = config.d_model // config.num_heads

        self.q_proj = nn.Linear(config.d_model, config.d_model)
        self.k_proj = nn.Linear(config.d_model, config.d_model)
        self.v_proj = nn.Linear(config.d_model, config.d_model)

        self.out_proj = nn.Linear(config.d_model, config.d_model)

    def forward(self, x_q, x_k, x_v, cos_q, sin_q, cos_k, sin_k, key_padding_mask=None):
        """
        x_q: [Batch, T_Query, Dim]
        x_k: [Batch, T_Key, Dim]
        x_v: [Batch, T_Key, Dim] -> Ist bei LocalAttn meist identisch mit x_k
        cos_q, sin_q: RoPE für Query-Länge
        cos_k, sin_k: RoPE für Key-Länge
        key_padding_mask: [Batch, T_Key] (True = Ignorieren / Padding)
        """
        
        B, T_q, _ = x_q.shape
        _, T_k, _ = x_k.shape

        q = self.q_proj(x_q)
        k = self.k_proj(x_k)
        v = self.v_proj(x_v)

        # B. Aufteilen in Heads 
        # [Batch, Time, Dim] -> [Batch, Time, Heads, HeadDim] -> [Batch, Heads, Time, HeadDim]
        q = q.view(B, T_q, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T_k, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T_k, self.num_heads, self.head_dim).transpose(1, 2)

        q, k = apply_rotary_pos_emb(q, k, cos_q, sin_q, cos_k, sin_k)
        attn_mask = None
        if key_padding_mask is not None:
            # PyTorch scaled_dot_product_attention erwartet bei Bool-Masken:
            # True = Element wird ignoriert (bei attn_mask)
            # Wir müssen sicherstellen, dass die Dimensionen broadcastbar sind
            # Mask Shape ist [Batch, T_Key]. Wir brauchen [Batch, 1, 1, T_Key] für (Batch, Head, Q-Time, K-Time)
            attn_mask = key_padding_mask.view(B, 1, 1, T_k)
            attn_mask = attn_mask.expand(-1, self.num_heads, T_q, -1)

        output = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        output = output.transpose(1, 2).contiguous().view(B, T_q, self.d_model)
        return self.out_proj(output)

class ActionBERTConfig:
    def __init__(self,
                 total_classes= 10,
                 input_dim=2048,
                 d_model=128,
                 num_heads=8,
                 num_layers=4,
                 ffn_dim=128,
                 dropout=0.05):
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
        self.norm = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(0.05)
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

        attn_mask = ~pk 
        all_masked_rows = attn_mask.all(dim=-1)
        
        if all_masked_rows.any():
            attn_mask[all_masked_rows, 0] = False 
            
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
        self.dropout = nn.Dropout(0.05)
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
    
        attn_output = self.rotary_mha(x_q=q, 
            x_k=k, 
            x_v=v, 
            cos_q=cos_windows, 
            sin_q=sin_windows, 
            cos_k=cos_windows, 
            sin_k=sin_windows, 
            key_padding_mask=~pk)
        
        attn_output = attn_output.view(B, self.window_dilation,-1, C)
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous().view(B, -1, C)
        attn_output = attn_output[:,:T,:]
        if torch.isnan(attn_output).any():
            raise Exception("NaN values detected in GLOBAL attention output")
        return attn_output.contiguous()
    
class Block(nn.Module):
    def __init__(self,config: ActionBERTConfig):
        super().__init__()
        self.local_attn = LocalAttention(config=config,local_window_size=64)
        self.global_attn = GlobalAttention(config=config,window_dilation=16)
        
        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)
        self.norm3 = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(0.05)
        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, config.d_model * 4),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(config.d_model * 4, config.d_model),
        )
    def forward(self, x,padding_mask,cos=None,sin=None):
        """
        x: [B, T, D]
        """
        attn_input = self.norm1(x)
        local_out = self.local_attn(attn_input, padding_mask=padding_mask, cos=cos, sin=sin)
        x = x + self.dropout(local_out) 
        global_input = self.norm2(x)
        global_out = self.global_attn(global_input, padding_mask,cos=cos,sin=sin)
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
        
        self.rotary_emb = RotaryEmbedding(config.d_model // config.num_heads)

        self.masking_module = MaskingModule(config.input_dim,config.d_model)

        self.layers = nn.ModuleList([
            Block(config)
            for _ in range(config.num_layers)
        ])

    def forward(self, x,patch_mask,padding_mask):
        seq_len = x.shape[1]
        cos, sin = self.rotary_emb(x, seq_len=seq_len)
        
        x = self.input_proj(x) 
        x = self.masking_module(x,mask=patch_mask)
        for layer in self.layers:
            x = layer(x,padding_mask=padding_mask,cos=cos,sin=sin)
            
        recon_features = self.output_head(x)
        prototype_logits = self.prototypes(x)
        boundary_logits = self.boundary_head(x)
        return recon_features, prototype_logits,     boundary_logits 

