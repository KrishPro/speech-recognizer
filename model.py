"""
Written by KrishPro @ KP

filename: `model.py`
"""

from typing import Callable, List
import torch.nn as nn
import numpy as np
import torch


def self_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, attn_mask: torch.Tensor = None) -> torch.Tensor:
    """
    Q.shape: (B*H, S, E/H)
    K.shape: (B*H, S, E/H)
    V.shape: (B*H, S, E/H)

    returns: (B*H, S, E/H)
    """
    B, S, E = Q.shape

    if not attn_mask: attn_mask = torch.zeros(S, S)

    energy: torch.Tensor = torch.nan_to_num(torch.softmax((torch.baddbmm(attn_mask, Q, K.mT) / (E ** 0.5)), dim=2))

    return torch.bmm(energy, V)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model:int, n_heads:int) -> None:
        super().__init__()

        self.Q: Callable[[torch.Tensor], torch.Tensor] = nn.Linear(d_model, d_model)
        self.K: Callable[[torch.Tensor], torch.Tensor] = nn.Linear(d_model, d_model)
        self.V: Callable[[torch.Tensor], torch.Tensor] = nn.Linear(d_model, d_model)

        self.out = nn.Linear(d_model, d_model)
        self.n_heads = n_heads

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None) -> torch.Tensor:
        """
        x.shape: (B, S, E)

        returns: (B, S, E)
        """
        B, S, E, H = *x.shape, self.n_heads

        Q: torch.Tensor = self.Q(x).reshape(B, S, H, E//H).permute(0, 2, 1, 3).reshape(B*H, S, E//H)
        K: torch.Tensor = self.K(x).reshape(B, S, H, E//H).permute(0, 2, 1, 3).reshape(B*H, S, E//H)
        V: torch.Tensor = self.V(x).reshape(B, S, H, E//H).permute(0, 2, 1, 3).reshape(B*H, S, E//H)

        out = self_attention(Q, K, V, attn_mask=attn_mask).reshape(B, H, S, E//H).permute(0, 2, 1, 3).reshape(B, S, E)

        return self.out(out)

class TransformerLayer(nn.Module):
    def __init__(self, d_model:int, n_heads:int, dim_feedforward:int, dropout_p:float) -> None:
        super().__init__()

        self.mha = MultiHeadAttention(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)

        self.feedforward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None) -> torch.Tensor:
        """
        x.shape: (B, S, E)

        returns: (B, S, E)
        """

        x = self.norm1(x + self.dropout(self.mha(x, attn_mask=attn_mask)))

        x = self.norm2(x + self.dropout(self.feedforward(x)))

        return x

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model:int, dropout_p:float=0.1, max_len:float=5000):
        super().__init__()
        self.d_model = d_model
        self.register_buffer('pos_embeddings', self.generate_sinusoids(max_len, d_model), persistent=False)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, embeddings: torch.Tensor):
        return self.dropout((embeddings * (self.d_model ** 0.5)) + self.pos_embeddings[:embeddings.size(1)])

    def generate_sinusoids(self, length, channels, max_timescale=10000):
        """Returns sinusoids for positional embedding"""
        assert channels % 2 == 0
        log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
        inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
        scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
        return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)


class Transformer(nn.Module):
    def __init__(self, n_mels:int, d_model:int, n_heads:int, dim_feedforward:int, n_layers:int, vocab_size:int, dropout_p:float, max_len:int=1024):
        super().__init__()

        self.input: Callable[[torch.Tensor], torch.Tensor] = nn.Sequential(
            nn.Conv1d(n_mels, d_model, kernel_size=9, padding=4),
            nn.Dropout(dropout_p)
        )

        self.pos_embed = PositionalEmbedding(d_model, dropout_p, max_len=max_len)

        self.layers: List[Callable[[torch.Tensor], torch.Tensor]] = nn.ModuleList([ TransformerLayer(d_model, n_heads, dim_feedforward) for _ in range(n_layers) ])

        self.output: Callable[[torch.Tensor], torch.Tensor] = nn.Linear(d_model, vocab_size)

        self.mask_token = -1e-25
        self.attn_mask = torch.empty(max_len, max_len).fill_(self.mask_token).triu_(1)

    def forward(self, x: torch.Tensor):
        """
        x.shape: (B, M, S)

        returns: (B, S, V)
        """
        S = x.size(2)

        x = self.pos_embed(self.input(x).transpose(1, 2))

        attn_mask = self.attn_mask[:S, :S]

        for layer in self.layers:
            x = layer(x, attn_mask=attn_mask)

        x = self.output(x)

        return x