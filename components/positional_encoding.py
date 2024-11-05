import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, sequence_length: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.sequence_length = sequence_length
        self.dropout = nn.Dropout(dropout)

        encoding = torch.zeros(sequence_length, d_model)
        position = torch.arange(0, sequence_length, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.00) / d_model))
        encoding[:, 0::2] = torch.sin(position * div)
        encoding[:, 1::2] = torch.cos(position * div)

        self.encoding = encoding.unsqueeze(0) 
        self.register_buffer('positional_encoding', encoding)
    
    def forward(self, x: torch.Tensor):
        x = x + (self.encoding[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)
    
