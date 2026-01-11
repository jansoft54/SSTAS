


from model.attention import GlobalAttention, LocalAttention, RotaryEmbedding
from model.bert import ActionBERTConfig, MaskingModule, RefinementBlock
import torch
import torch.nn as nn
import torch.nn.functional as F



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
    def apply_norm(self, norm_layer, x):
        x = norm_layer(x)
        return x 

    def forward(self, x, padding_mask, cos=None, sin=None):
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
        
        
class ActionProgressHead(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, kernel_size=3):
        super().__init__()
        
        self.head = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size, padding=kernel_size//2),
            nn.GroupNorm(4, hidden_dim),
            nn.ReLU(),
            
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=kernel_size//2),
            nn.GroupNorm(4, hidden_dim),
            nn.ReLU(),
            
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=kernel_size//2),
            nn.GroupNorm(4, hidden_dim),
            nn.ReLU(),

            nn.Conv1d(hidden_dim, 2, kernel_size=1) 
        )

    def forward(self, x):
        # x: [B, C, T]
        
        return self.head(x.transpose(1,2))

    
    
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

class ActionBoundary(nn.Module):
    def __init__(self,
                 config: ActionBERTConfig,
                 train_for_knowns=True
                 ):
        super().__init__()
        self.config = config
        self.train_for_knowns = train_for_knowns

        self.input_proj = nn.Linear(config.input_dim, config.d_model)

        num_centers = config.known_classes 

        self.prototypes = nn.Linear(
            config.d_model , num_centers, bias=False)
       
        self.final_norm = nn.LayerNorm(config.d_model)
        
       
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
            dim=num_centers, 
            num_f_maps=64,
           )

        self.register_buffer("class_centers", torch.zeros(
            num_centers, config.d_model ))
        self.register_buffer("centers_initialized", torch.zeros(
            num_centers, dtype=torch.bool))

        self.center_momentum = 0.99
        self.input_aug = InputAugmentation()
        
        self.action_progress_head = ActionProgressHead(input_dim=config.d_model)

    def _get_prototype_logits(self, x_, m, prototypes):
        sum_x = (x_ * m).sum(dim=1, keepdim=True)
        count = m.sum(dim=1, keepdim=True)  
        video_mean = sum_x / (count + 1e-6)
        x = (x_ - video_mean) * m
        
        x_norm = F.normalize(x, p=2, dim=-1)
        w_norm = F.normalize(prototypes.weight, p=2, dim=-1)
        prototype_logits = torch.matmul(
            x_norm, w_norm.t())  

        prototype_logits = prototype_logits * 16.0

        return prototype_logits, x_

    def forward(self, input, padding_mask, _run_name=None):
        input =  self.input_aug(input, padding_mask)
        seq_len = input.shape[1]
        cos, sin = None,None

        x = self.input_proj(input)
        x = x * padding_mask.unsqueeze(-1).type_as(x)

       
        for layer in self.stage_one_layers:
            x = layer(x, padding_mask=padding_mask, cos=cos, sin=sin)

        x = self.final_norm(x)

        m = padding_mask.unsqueeze(-1).type_as(x)

        prototype_logits, x = self._get_prototype_logits(x, m, self.prototypes)
        
        action_progress = self.action_progress_head(x)

        refine_logits = prototype_logits * m

        refine_logits, stages_output_logits = self.refinement_block(
            refine_logits, padding_mask)

        result = {
                  "prototype_logits": prototype_logits,
                  "refine_logits": refine_logits,
                  "stages_output_logits": stages_output_logits,
                  "action_progress": action_progress,
                  "embeddings": x,
                  }
        return result
