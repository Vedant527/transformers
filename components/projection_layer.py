import torch
import torch.nn as nn

class ProjectionLayer(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)
    
    def forward(self, x: torch.Tensor):
        return torch.log_softmax(self.proj(x), dim=-1)
