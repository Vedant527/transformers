import torch
import torch.nn as nn

from components.layer_normalization import LayerNormalization

class ResidualConnection(nn.Module):
    def __init__(self, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()
    
    def forward(self, x: torch.Tensor, sublayer: torch.Tensor):
        return x + self.dropout(sublayer(self.norm(x)))