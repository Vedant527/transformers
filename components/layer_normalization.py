import torch
import torch.nn as nn

class LayerNormalization(nn.Module):
    def __init__(self, eps: float = 10**-6):
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.ones(1))
    
    def forward(self, x: torch.Tensor):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.alpha * ((x - mean) / torch.sqrt(std + self.eps)) + self.beta