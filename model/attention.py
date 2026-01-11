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




class LocalAttention(nn.Module):
    def __init__(self, config, local_window_size=64):

        super().__init__()
        self.local_window_size = local_window_size
        self.norm = nn.LayerNorm(config.d_model)
        self.rotary_mha = RotaryMHA(config=config)
        self.local_rope = RotaryEmbedding(config.d_model // config.num_heads)

    def create_windows(self, x, padding_mask, cos, sin):
        B, T, C = x.shape
        W = self.local_window_size

        pad_end = (W - (T % W)) % W
        clean_mask = padding_mask.unsqueeze(-1).type_as(x)
        x = x * clean_mask

        if pad_end > 0:
            x_padded = F.pad(x, (0, 0, 0, pad_end))

            padding_mask_padded = F.pad(
                padding_mask, (0, pad_end), value=False)

        else:
            x_padded = x
            padding_mask_padded = padding_mask

        q_windows = x_padded.unfold(1, W, W).permute(
            0, 1, 3, 2).reshape(-1, W, C)

        """ cos_q = cos_padded.unfold(1, W, W).permute(
            0, 1, 3, 2).reshape(-1, W, cos_padded.shape[-1]).unsqueeze(1)
        """

        """ sin_q = sin_padded.unfold(1, W, W).permute(
            0, 1, 3, 2).reshape(-1, W, sin_padded.shape[-1]).unsqueeze(1)
        """

        half_pad = W // 2
        x_context = F.pad(x_padded, (0, 0, half_pad, half_pad))

        k_windows = x_context.unfold(
            1, 2*W, W).permute(0, 1, 3, 2).reshape(-1, 2*W, C)

        cos_k, sin_k = self.local_rope(k_windows, seq_len=2*W)
        cos_q = cos_k[:, :, half_pad: half_pad + W, :]
        sin_q = sin_k[:, :, half_pad: half_pad + W, :]

        padding_mask_k = F.pad(padding_mask_padded,
                               (half_pad, half_pad), value=False)
        padding_mask_k = padding_mask_k.unfold(1, 2*W, W).reshape(-1, 2*W)

        q = q_windows
        k = k_windows
        pk = padding_mask_k

        return q, k, pk, cos_q, sin_q, cos_k, sin_k

    def forward(self, x, padding_mask, cos=None, sin=None):
        batch_size, seq_len, dim = x.size()
        q, k, pk, cos_q, sin_q, cos_k, sin_k = self.create_windows(
            x, padding_mask=padding_mask, cos=cos, sin=sin)
        v = k

        attn_mask = pk.clone()
        rows_with_no_data = ~attn_mask.any(dim=-1)

        if rows_with_no_data.any():
            attn_mask[rows_with_no_data, 0] = True

        attn_output = self.rotary_mha(
            q, k, v, cos_q, sin_q, cos_k, sin_k, key_padding_mask=attn_mask)

        attn_output = attn_output.view(
            batch_size, -1, self.local_window_size, dim)

        attn_output = attn_output.flatten(1, 2)[:, :seq_len, :]
        if torch.isnan(attn_output).any():
            raise Exception("NaN values detected in LOCAL attention output")

        return attn_output.contiguous()


class GlobalAttention(nn.Module):
    def __init__(self, config, window_dilation=8):
        super().__init__()
        self.rotary_mha = RotaryMHA(config=config)
        self.norm = nn.LayerNorm(config.d_model)
        self.window_dilation = window_dilation
        self.global_rope = RotaryEmbedding(config.d_model // config.num_heads,)

    def create_windows(self, x, padding_mask, cos, sin):
        B, T, C = x.shape
        W = self.window_dilation
        pad_end = (W - (T % W)) % W

        """ cos_t = cos.squeeze(0).squeeze(0).unsqueeze(
            0).expand(B, -1, -1)  # [B, T, HeadDim]
        sin_t = sin.squeeze(0).squeeze(0).unsqueeze(0).expand(B, -1, -1)
        """
        if pad_end > 0:
            x_padded = F.pad(x, (0, 0, 0, pad_end))
            padding_mask_padded = F.pad(
                padding_mask, (0, pad_end), value=False)
           # cos_padded = F.pad(cos_t, (0, 0, 0, pad_end))
            #sin_padded = F.pad(sin_t, (0, 0, 0, pad_end))
        else:
            x_padded = x
            padding_mask_padded = padding_mask
           # cos_padded = cos_t
            #sin_padded = sin_t


        windows = x_padded.unfold(1, W, W).permute(0, 3, 1, 2).contiguous()
        windows = windows.flatten(0, 1)
        seq_len_global = windows.shape[1]
        
        
        cos_new, sin_new = self.global_rope(windows, seq_len=seq_len_global)

        padding_mask_padded = padding_mask_padded.unfold(
            1, W, W).transpose(1, 2).contiguous()
        padding_mask_padded = padding_mask_padded.flatten(0, 1)

        """cos_windows = cos_padded.unfold(
            1, W, W).permute(0, 3, 1, 2).contiguous()
        cos_windows = cos_windows.flatten(0, 1).unsqueeze(1)"""

        """sin_windows = sin_padded.unfold(
            1, W, W).permute(0, 3, 1, 2).contiguous()
        sin_windows = sin_windows.flatten(0, 1).unsqueeze(1)"""

        q, k, v = windows, windows, windows
        pk = padding_mask_padded
        return q, k, v, pk, cos_new, sin_new

    def forward(self, x, padding_mask, cos=None, sin=None):
        B, T, C = x.shape

        q, k, v, pk, cos_windows, sin_windows = self.create_windows(
            x, padding_mask=padding_mask, cos=cos, sin=sin)

        attn_mask = pk
        attn_output = self.rotary_mha(x_q=q,
                                      x_k=k,
                                      x_v=v,
                                      cos_q=cos_windows,
                                      sin_q=sin_windows,
                                      cos_k=cos_windows,
                                      sin_k=sin_windows,
                                      key_padding_mask=attn_mask)

        attn_output = attn_output.view(B, self.window_dilation, -1, C)
        attn_output = attn_output.permute(
            0, 2, 1, 3).contiguous().view(B, -1, C)
        attn_output = attn_output[:, :T, :]
        if torch.isnan(attn_output).any():
            raise Exception("NaN values detected in GLOBAL attention output")
        return attn_output.contiguous()

