import torch
import torch.nn as nn

from components.feed_fwd import FeedForward
from components.layer_normalization import LayerNormalization
from components.multihead_attention import MultiheadAttention
from components.residual_connection import ResidualConnection

class DecoderBlock(nn.Module):
    def __init__(self, self_attention_block: MultiheadAttention, 
                 cross_attention_block: MultiheadAttention, 
                 feed_forward_block: FeedForward,
                 dropout: float):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])

    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor, src_mask: torch.Tensor, tgt_mast: torch.Tensor):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mast))
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x

class Decoder(nn.Module):
    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()
    
    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor, src_mask: torch.Tensor, tgt_mast: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mast)
        return self.norm(x)


    
