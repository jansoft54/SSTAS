from model.util import MultiStageModel
import torch
import numpy as np

import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from dataclasses import dataclass
from model.attention import RotaryEmbedding, rotate_half, apply_rotary_pos_emb, RotaryMHA



_debug = {
    "a_unpad": [],
    "b_single_padded": [],
    "b_pure": [],
    "batched": [],
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
        if d.shape[0] > c.shape[0] and d.shape[0] == 2:  # Annahme: Batch Size 2
            d = d[1:2]  # Nimm das zweite Element (Video B)

        # Fall 2: Geflattet [Batch*X, Y, ...] -> Wir nehmen die zweite Hälfte
        elif d.shape[0] == 2 * c.shape[0]:
            half = c.shape[0]
            d = d[half:]  # Nimm die zweite Hälfte

        # Wenn Shapes hiernach nicht matchen, ist es komplex (z.B. Interleaving),
        # aber wir versuchen es mit Slicing weiter unten.

    # --- AUTOMATISCHES ZUSCHNEIDEN (Slicing) ---
    # Wir machen 'd' so klein wie 'c'

    slices = []
    possible = True

    if d.ndim != c.ndim:
        print(
            f"⚠️ {name}: Dimension Mismatch! {c.shape} vs {d.shape}. Kann nicht vergleichen.")
        return 999.9

    for dim in range(c.ndim):
        target_size = c.shape[dim]
        current_size = d.shape[dim]

        if current_size == target_size:
            slices.append(slice(None))  # Alles behalten
        elif current_size > target_size:
            # Padded Tensor ist größer -> Abschneiden!
            # Wir nehmen an, dass Daten immer am Anfang stehen (Index 0..N)
            slices.append(slice(0, target_size))
        else:
            # Clean ist größer als Padded? Das sollte technisch nicht passieren.
            possible = False

    if not possible:
        print(
            f"⚠️ {name}: Clean Tensor ist größer als Dirty Tensor! {c.shape} vs {d.shape}")
        return 999.9

    # Anwenden des Slices
    d_cut = d[tuple(slices)]

    # Differenz berechnen
    diff = (c - d_cut).abs().max().item()
    return diff


def addBreakpoint(name, value):
    _debug[_current_run_name].append(
        {"name": name, "value": value.detach().clone()})


def checkLeakage():
    L_a = 11000
    L_b = 9000
    assert len(_debug["b_single_padded"]) == len(_debug["b_pure"]) and len(
        _debug["b_pure"]) == len(_debug["batched"])

    for i in range(len(_debug["batched"])):
        b_single_pad = _debug["b_single_padded"][i]["value"]
        b_clean = _debug["b_pure"][i]["value"]

        batch = _debug["batched"][i]["value"]

        name = _debug["batched"][i]["name"]
        print(b_clean.shape, b_single_pad.shape)
        diff_padding = smart_compare(
            name, b_clean, b_single_pad, mode="padding")

        # diff_batch = (b_single_pad - batch[1:2]).abs().max().item()
        print(diff_padding)
       # print((b_clean[:,:L_b,:] - b_single_pad[:,:L_b,:]).abs().max().item())
        if diff_padding > 1e-4:  # or diff_batch > 1e-4:

            print(f"❌ LEAKAGE AT CHECKPOINT: {name}")

        else:
            print(f"✅ CHECKPOINT { name} WAS FINE")


@dataclass
class ActionBERTConfig:
    known_classes: int = 10
    input_dim: int = 2048
    d_model: int = 128
    num_heads: int = 8
    num_layers: int = 4
    ffn_dim: int = 128
    dropout: float = 0
    local_window_size: int = 128
    window_dilation: int = 64


class DilatedConv(nn.Module):
    def __init__(self, n_channels: int, dilation: int, kernel_size: int = 3):
        super().__init__()
        self.dilated_conv = nn.Conv1d(n_channels,
                                      n_channels,
                                      kernel_size=kernel_size,
                                      padding=dilation,
                                      dilation=dilation,
                                      )
        self.activation = nn.GELU()

    def forward(self, x, masks):
        """

        :param x:
        :param masks:
        :return:
        """
        
        return self.activation(self.dilated_conv(x.transpose(1,2))).transpose(1,2) * masks.unsqueeze(-1)
class MaskingModule(nn.Module):
    def __init__(self, input_dim, model_dim):
        super().__init__()
        self.input_dim = input_dim
        self.mask_value = model_dim
        self.mask_token = nn.Parameter(torch.randn(1, 1, model_dim) * 0.02)

    def forward(self, features, mask):
        """
        features: [B, T, D]
        mask: [B, T] BoolTensor (True = Position wurde zum Maskieren ausgewählt)
        """
        # Wir arbeiten auf einer Kopie, damit das Original für den Loss erhalten bleibt
        out = features.clone()

        if mask is None:
            return out

        # Zufallszahlen für jede Position im Batch generieren [B, T]
        probs = torch.rand(features.shape[:2], device=features.device)

        # --- FALL 1: 80% -> [MASK] Token ---
        # Wo maskiert werden soll UND Zufall < 0.8
        replace_mask = mask & (probs < 0.8)

        # Wir müssen den Token in die richtige Form broadcasten
        # token: [1, 1, D] -> wird automatisch auf [N_replace, D] broadcasted
        out[replace_mask] = self.mask_token.type_as(out)

        # --- FALL 2: 10% -> Random Noise ---
        # Wo maskiert werden soll UND 0.8 <= Zufall < 0.9
        random_mask = mask & (probs >= 0.8) & (probs < 0.9)

        # Erstelle Rauschen mit der gleichen Statistik (Mean/Std) wie die Features
        # Oder einfach Standard-Normalverteilung (oft gut genug bei LayerNorm)
        noise = torch.randn_like(features[random_mask])
        out[random_mask] = noise

        # --- FALL 3: 10% -> Original behalten (Identity) ---
        # Wo maskiert werden soll UND Zufall >= 0.9
        # Da wir 'out' als Klon von 'features' gestartet haben,
        # steht hier automatisch schon das Original drin.
        # Wir müssen also nichts tun!

        return out


"""
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
"""


class LocalAttention(nn.Module):
    def __init__(self, config: ActionBERTConfig, local_window_size=64):

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
    def __init__(self, config: ActionBERTConfig, window_dilation=8):
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


class GatedFusion(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.gate_net = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.Sigmoid()
        )
        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, x_local, x_global):
        combined = torch.cat([x_local, x_global], dim=-1)
        alpha = self.gate_net(combined)

        fused = alpha * x_global

        return self.out_norm(fused)


class RefinementBlock(nn.Module):
    def __init__(self, config, num_stages=4, num_layers=8, num_classes=None, dim=None, num_f_maps=64, detach=False):
        super().__init__()
        self.detach = detach
        self.refiner = MultiStageModel(
            num_stages=num_stages,
            num_layers=num_layers,
            num_f_maps=num_f_maps,
            num_classes=num_classes,
            dim=dim)

    def forward(self, logits, mask):
        if self.detach:
            logits = logits.detach()

        s2_input = logits.transpose(1, 2)
        m = mask.unsqueeze(1).type_as(s2_input)
        s2_input = s2_input * m
        stages_output = self.refiner(s2_input, m)

        # 4. Wir brauchen für den Dice/Focal Loss meistens alle Stages,
        # aber für den normalen Flow nehmen wir die LETZTE Stage [-1]
        last_stage_logits = stages_output[-1]  # [B, C, T]
        stages_output = stages_output.transpose(2, 3)
        # Zurück zu [B, T, C]
        return last_stage_logits.transpose(1, 2), stages_output


import torch
import torch.nn as nn


class Block(nn.Module):
    def __init__(self, config: ActionBERTConfig, dilation=1):
        super().__init__()
        self.local_attn = LocalAttention(
            config=config, local_window_size=config.local_window_size)
        self.global_attn = GlobalAttention(
            config=config, window_dilation=config.window_dilation)

        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)
        self.norm3 = nn.LayerNorm(config.d_model)

        self.dropout = nn.Dropout(0)

        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, config.d_model ),
           
          

        )
        self.dilated_conv = DilatedConv(
            n_channels=config.d_model, dilation=dilation, kernel_size=3)

    def apply_norm(self, norm_layer, x):
        """Hilfsfunktion für InstanceNorm auf [B, T, D]"""
        # [B, T, D] -> [B, D, T]
       # x = x.transpose(1, 2)
        x = norm_layer(x)
        # [B, D, T] -> [B, T, D]
        return x   #.transpose(1, 2)

    def forward(self, x, padding_mask, cos=None, sin=None):
        """
        x: [B, T, D]
        """
        B, T, D = x.shape
        mask = padding_mask.unsqueeze(-1).type_as(x)
        x = x * mask

        resid = x
        x_norm = self.apply_norm(self.norm1, x)
        local_out = self.local_attn(x_norm, padding_mask, cos, sin)
        x = resid + local_out
        x = x * mask
        resid = x
        x_norm = self.apply_norm(self.norm2, x)
        global_out = self.global_attn(x_norm, padding_mask, cos, sin)
        x = resid + global_out
        x = x * mask
        resid = x
        x_norm = self.apply_norm(self.norm3, x)
        ffn_out = self.ffn(x_norm)
        x = resid + self.dropout(ffn_out)
        x = x * mask
        return x
        
        
        
        """inputs = x 
        mask = padding_mask.unsqueeze(-1).type_as(x)

        # 1. Die Basis schaffen (Dilated Conv)
        #out = self.dilated_conv(x, padding_mask) 
        
        # 2. Lokale Verfeinerung (Addiert auf Conv-Output)
        resid_conv = inputs
        out = self.apply_norm(self.norm1, inputs)
        out = self.local_attn(out, padding_mask, cos, sin)
        out = out + resid_conv # <--- Wichtig: + out im Original

        # 3. Globale Verfeinerung (Addiert auf Local-Output)
        resid_local = out
        out = self.apply_norm(self.norm2, out)
        out = self.global_attn(out, padding_mask, cos, sin)
        out = out + resid_local # <--- Wichtig: + out im Original

        # 4. Finaler Block & Global Skip
        res_fin = out
        out = self.apply_norm(self.norm3, out)
        out = self.ffn(out)
        out = self.dropout(out)
        
        # Der finale "Global Skip" vom allerersten Anfang
        return (out + res_fin) * mask"""



class InputAugmentation(nn.Module):
    def __init__(self, noise_std=0.05, dropout_rate=0.3):
        super().__init__()
        self.noise_std = noise_std
        self.channel_dropout = nn.Dropout1d(p=0.3)

    def forward(self, x, padding_mask):
        """
        x: [B, T, D]
        padding_mask: [B, T]
        """
        if self.training:
            x = x.transpose(1, 2)
            x = self.channel_dropout(x)
            x = x.transpose(1, 2)

        return  x * padding_mask.unsqueeze(-1).type_as(x)

class ActionBERT(nn.Module):
    def __init__(self,
                 config: ActionBERTConfig,
                 train_for_knowns=True
                 ):
        super().__init__()
        self.config = config
        self.train_for_knowns = train_for_knowns

        self.input_proj = nn.Linear(config.input_dim, config.d_model)

        """self.output_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.d_model)
        )"""
        unknown_pseudo_class = 1
        num_centers = config.known_classes if train_for_knowns else config.known_classes + unknown_pseudo_class

        self.prototypes = nn.Linear(
            config.d_model // 4, num_centers, bias=False)
        self.prototypes_unk = nn.Linear(
            config.d_model // 4, num_centers, bias=False)

        self.final_norm = nn.LayerNorm(config.d_model)
        
        self.feature_purifier_enc = nn.Sequential(
        nn.Linear(config.d_model, config.d_model // 4),
        nn.GELU(),
        nn.LayerNorm(config.d_model // 4)
            )


        self.rotary_emb = RotaryEmbedding(config.d_model // config.num_heads)
        self.masking_module = MaskingModule(config.input_dim, config.d_model)

        self.stage_one_layers = nn.ModuleList([
            Block(config, dilation=2**i)
            for i in range(config.num_layers)
        ])

        self.refinement_block = RefinementBlock(
            config,
            num_classes=num_centers,
            num_stages=5,
            num_layers=12,
            dim=num_centers, #+ 32,
            num_f_maps=64,
           )

        self.register_buffer("class_centers", torch.zeros(
            num_centers, config.d_model // 4))
        self.register_buffer("centers_initialized", torch.zeros(
            num_centers, dtype=torch.bool))

        self.center_momentum = 0.99
        
        self.input_aug = InputAugmentation()

    def _get_prototype_logits(self, x, m, prototypes):
        sum_x = (x * m).sum(dim=1, keepdim=True)
        count = m.sum(dim=1, keepdim=True)  # [B, 1, 1]
        video_mean = sum_x / (count + 1e-6)
        x = (x - video_mean) * m
        
        x = self._purify_features(x, m) * m

        x_norm = F.normalize(x, p=2, dim=-1)
        w_norm = F.normalize(prototypes.weight, p=2, dim=-1)
        prototype_logits = torch.matmul(
            x_norm, w_norm.t())  # [B, T, NumClasses]

        prototype_logits = prototype_logits * 16.0

        return prototype_logits, x

    def _train_unk(self, input, patch_mask, padding_mask,):
        input =  self.input_aug(input, padding_mask)
        
        x_for_unk = self.input_proj(input)
        seq_len = input.shape[1]
        cos, sin = self.rotary_emb(input, seq_len=seq_len)
        x_for_unk = x_for_unk * padding_mask.unsqueeze(-1).type_as(x_for_unk)
        
        if self.training:
            x_for_unk = self.masking_module(x_for_unk, mask=patch_mask)

        for layer in self.stage_one_layers:
            x_for_unk = layer(
                x_for_unk, padding_mask=padding_mask, cos=cos, sin=sin)
        x_for_unk = self.final_norm(x_for_unk)
        
        m = padding_mask.unsqueeze(-1).type_as(x_for_unk)
        
        prototype_logits_unk, x_for_unk_centered = self._get_prototype_logits(
            x_for_unk, m, self.prototypes_unk)
        


        refine_logits_unk = prototype_logits_unk * m
      #  progress_pred = self.progress_head(x_for_unk_centered)
       # time_features = self.progress_expansion(progress_pred) 

      #  refine_input = torch.cat([refine_logits_unk, x_for_unk_centered], dim=-1)

        unkown_logits, stages_output_unkown_logits = self.refinement_block(
            refine_logits_unk, padding_mask)

        return prototype_logits_unk, unkown_logits, stages_output_unkown_logits, x_for_unk_centered,None
    
    def _purify_features(self, x_centered, padding_mask):
        if self.training:

            noise = torch.randn_like(x_centered) * 0.05 
            x_noisy = x_centered + noise
        else:
            x_noisy = x_centered

        purified_x = self.feature_purifier_enc(x_noisy)
      #  reconstructed_x = self.feature_purifier_dec(purified_x)
     
        return purified_x
    
    def _train_knowns(self, input, patch_mask, padding_mask,):
        input =  self.input_aug(input, padding_mask)
        seq_len = input.shape[1]
        cos, sin = self.rotary_emb(input, seq_len=seq_len)

        x = self.input_proj(input)
        

        x = x * padding_mask.unsqueeze(-1).type_as(x)

        recon_target = x.detach()

        if self.training:
            x = self.masking_module(x, mask=patch_mask)

        for layer in self.stage_one_layers:
            x = layer(x, padding_mask=padding_mask, cos=cos, sin=sin)

        x = self.final_norm(x)

        m = padding_mask.unsqueeze(-1).type_as(x)

        prototype_logits, x = self._get_prototype_logits(x, m, self.prototypes)

        refine_logits = prototype_logits * m

        refine_logits, stages_output_logits = self.refinement_block(
            refine_logits, padding_mask)
        return prototype_logits, refine_logits, stages_output_logits, x, None

    def forward(self, input, patch_mask, padding_mask, _run_name=None):
      #  global _current_run_name
       # _current_run_name = _run_name
        if self.train_for_knowns:
            prototype_logits, refine_logits, stages_output_logits, x,progress_pred = self._train_knowns(
                input, patch_mask, padding_mask)
        else:
            prototype_logits, refine_logits, stages_output_logits, x ,progress_pred= self._train_unk(
                input, patch_mask, padding_mask)

        result = {"recon_features": None,
                  "recon_target": None,
                  "prototype_logits": prototype_logits,
                  "refine_logits": refine_logits,
                  "stages_output_logits": stages_output_logits,
                  "progress_pred": progress_pred,
                  "embeddings": x,

                  }
        return result
