import torch.nn as nn
import torch.nn.functional as F

class SingleStageTCN(nn.Module):
    # Originally written by yabufarha
    # https://github.com/yabufarha/ms-tcn/blob/master/model.py

    def __init__(self, in_channel, n_features, n_classes, n_layers):
        super().__init__()
        self.conv_in = nn.Conv1d(in_channel, n_features, 1)
        layers = [
            DilatedResidualLayer(2**i, n_features, n_features) for i in range(n_layers)]
        self.layers = nn.ModuleList(layers)
        self.conv_out = nn.Conv1d(n_features, n_classes, 1)

    def forward(self, x):
        out = self.conv_in(x)
        for layer in self.layers:
            out = layer(out)
        out = self.conv_out(out)
        return out
    
class DilatedResidualLayer(nn.Module):
    # Originally written by yabufarha
    # https://github.com/yabufarha/ms-tcn/blob/master/model.py

    def __init__(self, dilation, in_channel, out_channels):
        super().__init__()
        self.conv_dilated = nn.Conv1d(
            in_channel, out_channels, 3, padding=dilation, dilation=dilation)
        self.conv_in = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout()

    def forward(self, x):
        out = F.relu(self.conv_dilated(x))
        out = self.conv_in(out)
        out = self.dropout(out)
        return x + out
