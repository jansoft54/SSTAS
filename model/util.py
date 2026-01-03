import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import copy
import numpy as np


class MultiStageModel(nn.Module):
    def __init__(self, num_stages, num_layers, num_f_maps, dim, num_classes):
        super(MultiStageModel, self).__init__()
        self.stage1 = SingleStageModel(
            num_layers, num_f_maps, dim, num_classes)
        self.stages = nn.ModuleList([copy.deepcopy(SingleStageModel(
            num_layers, num_f_maps, num_classes, num_classes)) for s in range(num_stages-1)])

    def forward(self, x, mask):
        out = self.stage1(x, mask)
        outputs = out.unsqueeze(0)
        for s in self.stages:
            out = s(F.softmax(out, dim=1) * mask, mask)
            outputs = torch.cat((outputs, out.unsqueeze(0)), dim=0)
        return outputs


class SingleStageModel(nn.Module):
    def __init__(self, num_layers, num_f_maps, dim, num_classes):
        super(SingleStageModel, self).__init__()
        self.conv_1x1 = nn.Conv1d(dim, num_f_maps, 1)
        self.layers = nn.ModuleList([copy.deepcopy(DilatedResidualLayer(
            2 ** i, num_f_maps, num_f_maps)) for i in range(num_layers)])
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x, mask):
        out = self.conv_1x1(x) * mask
        for layer in self.layers:
            out = layer(out, mask)
        out = self.conv_out(out) * mask
        return out


class DilatedResidualLayer(nn.Module):
    def __init__(self, dilation, in_channels, out_channels):
        super(DilatedResidualLayer, self).__init__()
        self.conv_dilated = nn.Conv1d(
            in_channels, out_channels, 3, padding=dilation, dilation=dilation)
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout()

    def forward(self, x, mask):
        x = x * mask
        out = F.relu(self.conv_dilated(x))
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return (x + out) * mask





class RefinementBlockTwo(nn.Module):
    def __init__(self, config, num_stages=3, num_layers=8,num_classes=None):
        super().__init__()
        self.num_classes = num_classes
        self.detach = False
        
        self.projector = nn.Linear(config.d_model, 32)
        
        import copy
        refine_config = copy.deepcopy(config)
        refine_config.d_model = 32
        
        self.stages = nn.ModuleList()
        self.stage_classifiers = nn.ModuleList()
        
        for s in range(num_stages):
            layers = nn.ModuleList([
                Block(refine_config, dilation=2**i) 
                for i in range(num_layers)
            ])
            self.stages.append(layers)
            
            self.stage_classifiers.append(nn.Linear(32, num_classes))

    def forward(self, logits, mask, cos=None, sin=None):
       
        if self.detach:
            logits = logits.detach()

        B, T, C = logits.shape
        
        x = self.projector(logits)
        
        all_stages_outputs = []
        
        for s in range(len(self.stages)):
            for block in self.stages[s]:
            
                x = block(x, mask, cos=cos, sin=sin)
            
            stage_logits = self.stage_classifiers[s](x) # [B, T, num_classes]
            
            stage_logits = stage_logits * mask.unsqueeze(-1)
            all_stages_outputs.append(stage_logits)
            
     
        last_stage_logits = all_stages_outputs[-1]
        stacked_output = torch.stack(all_stages_outputs, dim=0) 
        
        return last_stage_logits, stacked_output