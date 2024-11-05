import torch
import torch.nn as nn

class FeedForward(nn.Module):
    def __init__(self, d_model: int, hidden_layer_size: int, dropout: float):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, hidden_layer_size)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(hidden_layer_size, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))
