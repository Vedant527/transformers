import torch.nn as nn
from torch import Tensor

from components.feed_fwd import FeedForward
from components.layer_normalization import LayerNormalization
from components.multihead_attention import MultiheadAttention
from components.residual_connection import ResidualConnection

class EncoderBlock(nn.Module):
    def __init__(self, self_attention_block: MultiheadAttention, feed_forward_block: FeedForward, dropout: float):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.dropout = nn.Dropout(dropout)
        self.residual_connections = nn.ModuleList(ResidualConnection(dropout) for _ in range(2))
    
    def forward(self, x: Tensor, src_mask: Tensor) -> Tensor:
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        return self.residual_connections[1](x, self.feed_forward_block)

class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)