import torch
import numpy as np

import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from dataclasses import dataclass
from model.attention import RotaryEmbedding, rotate_half, apply_rotary_pos_emb, RotaryMHA



_debug = {
    "a_unpad":[],
    "b_single_padded":[],
    "b_pure":[],
    "batched":[],
}

_current_run_name = None

def smart_compare(name, clean_tensor, dirty_tensor, mode="padding"):
    """
    Vergleicht zwei Tensoren, auch wenn 'dirty_tensor' Padding oder Batch-Nachbarn enthält.
    Schneidet 'dirty_tensor' automatisch auf die Form von 'clean_tensor' zu.
    
    mode: 
      - "padding": Vergleicht 'b_pure' vs 'b_single_padded' (Nullen am Ende)
      - "batch": Vergleicht 'b_single_padded' vs 'batched' (Video A ist Nachbar)
    """
    c = clean_tensor.detach().cpu()
    d = dirty_tensor.detach().cpu()
    
    # --- SPEZIALFALL: BATCH EXTRAKTION ---
    if mode == "batch":
        # Wir müssen Video B aus dem Batch extrahieren.
        # Fall 1: Standard [Batch, Time, ...] -> Batch ist Dim 0
        if d.shape[0] > c.shape[0] and d.shape[0] == 2: # Annahme: Batch Size 2
            d = d[1:2] # Nimm das zweite Element (Video B)
            
        # Fall 2: Geflattet [Batch*X, Y, ...] -> Wir nehmen die zweite Hälfte
        elif d.shape[0] == 2 * c.shape[0]:
            half = c.shape[0]
            d = d[half:] # Nimm die zweite Hälfte
            
        # Wenn Shapes hiernach nicht matchen, ist es komplex (z.B. Interleaving),
        # aber wir versuchen es mit Slicing weiter unten.

    # --- AUTOMATISCHES ZUSCHNEIDEN (Slicing) ---
    # Wir machen 'd' so klein wie 'c'
    
    slices = []
    possible = True
    
    if d.ndim != c.ndim:
        print(f"⚠️ {name}: Dimension Mismatch! {c.shape} vs {d.shape}. Kann nicht vergleichen.")
        return 999.9

    for dim in range(c.ndim):
        target_size = c.shape[dim]
        current_size = d.shape[dim]
        
        if current_size == target_size:
            slices.append(slice(None)) # Alles behalten
        elif current_size > target_size:
            # Padded Tensor ist größer -> Abschneiden!
            # Wir nehmen an, dass Daten immer am Anfang stehen (Index 0..N)
            slices.append(slice(0, target_size))
        else:
            # Clean ist größer als Padded? Das sollte technisch nicht passieren.
            possible = False
            
    if not possible:
        print(f"⚠️ {name}: Clean Tensor ist größer als Dirty Tensor! {c.shape} vs {d.shape}")
        return 999.9

    # Anwenden des Slices
    d_cut = d[tuple(slices)]
    
    # Differenz berechnen
    diff = (c - d_cut).abs().max().item()
    return diff
def addBreakpoint(name,value):
    _debug[_current_run_name].append({"name":name,"value":value.detach().clone()})
    

def checkLeakage():
    L_a = 11000
    L_b = 9000
    assert len(_debug["b_single_padded"]) == len(_debug["b_pure"]) and len(_debug["b_pure"]) == len(_debug["batched"])
 
    for i in range(len(_debug["batched"])):
        b_single_pad = _debug["b_single_padded"][i]["value"]
        b_clean = _debug["b_pure"][i]["value"]

        batch = _debug["batched"][i]["value"]
       

        name = _debug["batched"][i]["name"]
        print(b_clean.shape,b_single_pad.shape)
        diff_padding = smart_compare(name, b_clean, b_single_pad, mode="padding")
  
        #diff_batch = (b_single_pad - batch[1:2]).abs().max().item()
        print(diff_padding)
       # print((b_clean[:,:L_b,:] - b_single_pad[:,:L_b,:]).abs().max().item())
        if diff_padding > 1e-4: #or diff_batch > 1e-4:
             
            print(f"❌ LEAKAGE AT CHECKPOINT: {name}")
            
        else:
            print(f"✅ CHECKPOINT { name} WAS FINE")
 

@dataclass
class ActionBERTConfig:
    total_classes: int = 10
    input_dim: int = 2048
    d_model: int = 128
    num_heads: int = 8
    num_layers: int = 4
    ffn_dim: int = 128
    dropout: float = 0
    local_window_size: int = 128
    window_dilation: int = 64


class MaskingModule(nn.Module):
    def __init__(self, input_dim, model_dim):
        super().__init__()
        self.input_dim = input_dim
        self.mask_value = model_dim
        self.mask_token = nn.Parameter(torch.randn(1, 1, model_dim) * 0.02)

    def forward(self, features, mask):
        features_masked = features.clone()
        token = self.mask_token.type_as(features_masked)

        if mask != None:
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
    def __init__(self, config: ActionBERTConfig, local_window_size=64):

        super().__init__()
        self.local_window_size = local_window_size
        self.norm = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.rotary_mha = RotaryMHA(config=config)

    def create_windows(self, x, padding_mask, cos, sin):
        B, T, C = x.shape
        W = self.local_window_size

        cos_t = cos.squeeze(0).squeeze(0)
        sin_t = sin.squeeze(0).squeeze(0)
        
        cos_t = cos_t.unsqueeze(0).expand(B, -1, -1)
        sin_t = sin_t.unsqueeze(0).expand(B, -1, -1)
        cos_t = cos_t.contiguous() 
        sin_t = sin_t.contiguous()
        pad_end = (W - (T % W)) % W
        clean_mask = padding_mask.unsqueeze(-1).type_as(x)
        x = x * clean_mask
        cos_t = cos_t * clean_mask
        sin_t = sin_t *clean_mask

        if pad_end > 0:
            x_padded = F.pad(x, (0, 0, 0, pad_end))
           

            padding_mask_padded = F.pad(
                padding_mask, (0, pad_end), value=False)

            cos_padded = F.pad(cos_t, (0, 0, 0, pad_end))
            sin_padded = F.pad(sin_t, (0, 0, 0, pad_end))
        else:
            x_padded = x
            padding_mask_padded = padding_mask
            cos_padded = cos_t
            sin_padded = sin_t
            

       # print("hi",x_padded.shape,x_padded.shape)

        
        
        q_windows = x_padded.unfold(1, W, W).permute(
            0, 1, 3, 2).reshape(-1, W, C)


        cos_q = cos_padded.unfold(1, W, W).permute(
            0, 1, 3, 2).reshape(-1, W, cos_padded.shape[-1]).unsqueeze(1)
        

        sin_q = sin_padded.unfold(1, W, W).permute(
            0, 1, 3, 2).reshape(-1, W, sin_padded.shape[-1]).unsqueeze(1)
        

        half_pad = W // 2
        x_context = F.pad(x_padded, (0, 0, half_pad, half_pad))

        cos_context = F.pad(cos_padded, (0, 0, half_pad, half_pad))
        sin_context = F.pad(sin_padded, (0, 0, half_pad, half_pad))

        k_windows = x_context.unfold(
            1, 2*W, W).permute(0, 1, 3, 2).reshape(-1, 2*W, C)
        

        padding_mask_k = F.pad(padding_mask_padded,
                               (half_pad, half_pad), value=False)
        padding_mask_k = padding_mask_k.unfold(1, 2*W, W).reshape(-1, 2*W)

        cos_k = cos_context.unfold(
            1, 2*W, W).permute(0, 1, 3, 2).reshape(-1, 2*W, cos_padded.shape[-1]).unsqueeze(1)
        sin_k = sin_context.unfold(
            1, 2*W, W).permute(0, 1, 3, 2).reshape(-1, 2*W, sin_padded.shape[-1]).unsqueeze(1)

      

        q = q_windows
        k = k_windows
        pk = padding_mask_k

        return q, k, pk, cos_q, sin_q, cos_k, sin_k

    def forward(self, x, padding_mask, cos=None, sin=None):
        batch_size, seq_len, dim = x.size()
        q, k, pk, cos_q, sin_q, cos_k, sin_k = self.create_windows(
            x, padding_mask=padding_mask, cos=cos, sin=sin)
        v = k
       
        """
        addBreakpoint("1",q)
        addBreakpoint("2",k)
        addBreakpoint("3",pk.float())
        addBreakpoint("4",cos_q)
        addBreakpoint("5",sin_q)
        addBreakpoint("6",cos_k)
        addBreakpoint("7",sin_k)"""
        
        
        
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
    def __init__(self, config: ActionBERTConfig, window_dilation=8):
        super().__init__()
        self.rotary_mha = RotaryMHA(config=config)
        self.norm = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.window_dilation = window_dilation

    def create_windows(self, x, padding_mask, cos, sin):
        B, T, C = x.shape
        W = self.window_dilation
        pad_end = (W - (T % W)) % W

        cos_t = cos.squeeze(0).squeeze(0).unsqueeze(
            0).expand(B, -1, -1)  # [B, T, HeadDim]
        sin_t = sin.squeeze(0).squeeze(0).unsqueeze(0).expand(B, -1, -1)

        if pad_end > 0:
            x_padded = F.pad(x, (0, 0, 0, pad_end))
            padding_mask_padded = F.pad(
                padding_mask, (0, pad_end), value=False)
            cos_padded = F.pad(cos_t, (0, 0, 0, pad_end))
            sin_padded = F.pad(sin_t, (0, 0, 0, pad_end))
        else:
            x_padded = x
            padding_mask_padded = padding_mask
            cos_padded = cos_t
            sin_padded = sin_t

        windows = x_padded.unfold(1, W, W).permute(0, 3, 1, 2).contiguous()
        windows = windows.flatten(0, 1)

        padding_mask_padded = padding_mask_padded.unfold(
            1, W, W).transpose(1, 2).contiguous()
        padding_mask_padded = padding_mask_padded.flatten(0, 1)

        cos_windows = cos_padded.unfold(
            1, W, W).permute(0, 3, 1, 2).contiguous()
        cos_windows = cos_windows.flatten(0, 1).unsqueeze(1)

        sin_windows = sin_padded.unfold(
            1, W, W).permute(0, 3, 1, 2).contiguous()
        sin_windows = sin_windows.flatten(0, 1).unsqueeze(1)

        q, k, v = windows, windows, windows
        pk = padding_mask_padded
        return q, k, v, pk, cos_windows, sin_windows

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


class GatedFusion(nn.Module):
    def __init__(self, d_model):
        super().__init__()
    
        
        self.gate_net = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(), 
            
            nn.Linear(d_model, d_model),
            
            # 3. Sigmoid für 0..1 Range
            nn.Sigmoid() 
        )
        self.out_norm = nn.LayerNorm(d_model)


    def forward(self, x_local, x_global):
        combined = torch.cat([x_local, x_global], dim=-1)
        alpha = self.gate_net(combined)
        
     
        fused = (alpha * x_local) + ((1 - alpha) * x_global)
        
        return self.out_norm(fused)
class Block(nn.Module):
    def __init__(self, config: ActionBERTConfig):
        super().__init__()
        self.local_attn = LocalAttention(
            config=config, local_window_size=config.local_window_size)
        self.global_attn = GlobalAttention(
            config=config, window_dilation=config.window_dilation)

        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)
        self.norm3 = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.gated_fusion = GatedFusion(d_model=config.d_model)
        
        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, config.d_model * 4),
            nn.GELU(),
            nn.Linear(config.d_model * 4, config.d_model),
        )

    def forward(self, x, padding_mask, cos=None, sin=None):
        """
        x: [B, T, D]
        """
        resid = x
        x_norm_local = self.norm1(x)
        local_out = self.local_attn(x_norm_local, padding_mask, cos, sin)
        
      #  x = resid + self.dropout(local_out)
        x_norm_global = self.norm2(x)
        global_out = self.global_attn(x_norm_global, padding_mask, cos, sin)
        fused_attn = self.gated_fusion(local_out, global_out)
        x = resid + self.dropout(fused_attn)
        
        #x = resid + self.dropout(global_out)

        resid = x
        x_norm_ffn = self.norm3(x)
        ffn_out = self.ffn(x_norm_ffn)
        x = resid + self.dropout(ffn_out)
        return x


class ActionBERT(nn.Module):
    def __init__(self,
                 config: ActionBERTConfig,
                 ):
        super().__init__()
        self.config = config
        self.input_proj = nn.Linear(config.input_dim, config.d_model)
        self.output_head = nn.Linear(config.d_model, config.input_dim)
        self.prototypes = nn.Linear(
            config.d_model, config.total_classes, bias=False)
        self.final_norm = nn.LayerNorm(config.d_model)

        self.rotary_emb = RotaryEmbedding(config.d_model // config.num_heads)
        self.masking_module = MaskingModule(config.input_dim, config.d_model)

        self.layers = nn.ModuleList([
            Block(config)
            for _ in range(config.num_layers)
        ])

    def forward(self, x, patch_mask, padding_mask,_run_name=None):
        global _current_run_name
        _current_run_name = _run_name
        seq_len = x.shape[1]
        cos, sin = self.rotary_emb(x, seq_len=seq_len)
      #  cos = torch.zeros_like(cos)
       # sin = torch.zeros_like(sin)
        x = self.input_proj(x)
       

        x = self.masking_module(x, mask=patch_mask)
       

        mask_float = padding_mask.unsqueeze(-1).type_as(x)

       # print(x[~padding_mask])
        for layer in self.layers:
            x = layer(x, padding_mask=padding_mask, cos=cos, sin=sin)
            
        #    x = x * mask_float
        #    print(x[~padding_mask])

        x = self.final_norm(x)
    #    x = x*mask_float

        recon_features = self.output_head(x)
        prototype_logits = self.prototypes(x)
        # = self.boundary_head(x)
        return recon_features, prototype_logits, x


