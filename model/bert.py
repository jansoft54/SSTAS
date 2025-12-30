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
        self.dropout = nn.Dropout(config.dropout)
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
        self.dropout = nn.Dropout(config.dropout)
        self.window_dilation = window_dilation
        self.global_rope = RotaryEmbedding(config.d_model // config.num_heads,)

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
        seq_len_global = windows.shape[1]
        cos_new, sin_new = self.global_rope(windows, seq_len=seq_len_global)

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
    def __init__(self, config, num_layers=8, num_classes=None, dim=None, detach=False):
        super().__init__()
        self.detach = detach
        self.refiner = MultiStageModel(
            num_stages=4,
            num_layers=num_layers,
            num_f_maps=64,
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

        self.dropout = nn.Dropout(config.dropout)

        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, config.d_model * 2),
            nn.GELU(),
            nn.Linear(config.d_model*2, config.d_model * 2),
            nn.GELU(),
            nn.Linear(config.d_model*2, config.d_model),

        )

    def apply_norm(self, norm_layer, x):
        """Hilfsfunktion für InstanceNorm auf [B, T, D]"""
        # [B, T, D] -> [B, D, T]
       # x = x.transpose(1, 2)
        x = norm_layer(x)
        # [B, D, T] -> [B, T, D]
        return x  # .transpose(1, 2)

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


class ActionBERT(nn.Module):
    def __init__(self,
                 config: ActionBERTConfig,
                 ):
        super().__init__()
        self.config = config

        self.input_proj = nn.Linear(config.input_dim, config.d_model)

        self.output_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.d_model)
        )

        self.prototypes = nn.Linear(
            config.d_model, config.total_classes, bias=False)
        self.final_norm = nn.LayerNorm(config.d_model)

        self.rotary_emb = RotaryEmbedding(config.d_model // config.num_heads)
        self.masking_module = MaskingModule(config.input_dim, config.d_model)

        self.stage_one_layers = nn.ModuleList([
            Block(config, dilation=2**i)
            for i in range(config.num_layers)
        ])
        self.refinement_block = RefinementBlock(
            config,
            num_layers=12,
            num_classes=config.total_classes,
            dim=config.total_classes)
        self.unkown_detector = RefinementBlock(
            config,
            num_layers=12,
            num_classes=2,
            dim=config.d_model,
            detach=True)

        self.register_buffer("class_centers", torch.zeros(
            config.total_classes, config.d_model))
        self.register_buffer("centers_initialized", torch.zeros(
            config.total_classes, dtype=torch.bool))
        self.center_momentum = 0.99
        self.initial_in = nn.InstanceNorm1d(config.d_model, affine=True)

    def forward(self, x, patch_mask, padding_mask, _run_name=None):
        global _current_run_name
        _current_run_name = _run_name
        seq_len = x.shape[1]
        cos, sin = self.rotary_emb(x, seq_len=seq_len)

        x = self.input_proj(x)
        x = x * padding_mask.unsqueeze(-1).type_as(x)

        recon_target = x.detach()

        if self.training:
            x = self.masking_module(x, mask=patch_mask)

        for layer in self.stage_one_layers:
            x = layer(x, padding_mask=padding_mask, cos=cos, sin=sin)

        x = self.final_norm(x)
        m = padding_mask.unsqueeze(-1).type_as(x)

        sum_x = (x * m).sum(dim=1, keepdim=True)
        count = m.sum(dim=1, keepdim=True)  # [B, 1, 1]
        video_mean = sum_x / (count + 1e-6)
        x = (x - video_mean) * m

        x_norm = F.normalize(x, p=2, dim=-1)
        w_norm = F.normalize(self.prototypes.weight, p=2, dim=-1)
        prototype_logits = torch.matmul(
            x_norm, w_norm.t())  # [B, T, NumClasses]

        recon_features = self.output_head(x)
        prototype_logits = prototype_logits * 16.0

        refine_logits = prototype_logits * m

        unkown_logits, stages_output_unkown_logits = self.unkown_detector(
            x, padding_mask)

        refine_logits, stages_output_logits = self.refinement_block(
            refine_logits, padding_mask)

        result = {"recon_features": recon_features,
                  "recon_target": recon_target,
                  "prototype_logits": prototype_logits,
                  "refine_logits": refine_logits,
                  "stages_output_logits": stages_output_logits,
                  "unkown_logits": unkown_logits,
                  "stages_output_unknown_logits": stages_output_unkown_logits,
                  "embeddings": x
                  }
        return result
