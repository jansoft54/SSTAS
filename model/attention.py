import torch
import numpy as np
import torch.nn.functional as F
import torch
import torch.nn as nn

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

        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(config.d_model, config.d_model)
        self.k_proj = nn.Linear(config.d_model, config.d_model)
        self.v_proj = nn.Linear(config.d_model, config.d_model)

        self.out_proj = nn.Linear(config.d_model, config.d_model)
        self.dropout_p = config.dropout


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
        
        q = q.view(B, T_q, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T_k, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T_k, self.num_heads, self.head_dim).transpose(1, 2)
      
        q, k = apply_rotary_pos_emb(q, k, cos_q, sin_q, cos_k, sin_k)
        attn_mask = None
        if key_padding_mask is not None:

            attn_mask = key_padding_mask.view(B, 1, 1, T_k)            
            attn_mask = attn_mask.expand(-1, self.num_heads, T_q, -1)
        
        dropout_rate = self.dropout_p if self.training else 0.0

        output = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=dropout_rate)
        
        output = output.transpose(1, 2).contiguous().view(B, T_q, self.d_model)
        return self.out_proj(output)
