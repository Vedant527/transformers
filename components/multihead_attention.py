import torch
import torch.nn as nn
import math

class MultiheadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        assert d_model % num_heads == 0, 'Invalid number of heads, does not evenly split model dim'

        self.d_k = d_model / num_heads
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor, dropout: nn.Dropout):
        d_k = query.shape[-1]
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            attention_scores.masked_fill_(mask==0, -1e9)
        attention_scores = attention_scores.softmax(dim=-1)
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        return (attention_scores @ value), attention_scores


    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor):
        query = self.w_q(q) 
        key = self.w_k(k)
        value = self.w_v(v)

        query = query.view(query.shape[0], query.shape[1], self.num_heads, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.num_heads, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.num_heads, self.d_k).transpose(1, 2)

        x, self.attention_scores = MultiheadAttention.attention(query, key, value, mask, self.dropout)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.num_heads * self.d_k)
        return self.w_o(x)

