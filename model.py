"""
Written by KrishPro @ KP
filename: `model.py`
"""

from typing import Callable, List
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch


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

def self_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, attn_mask: torch.Tensor = None):
    """
    Q.shape: (B, S, E)
    K.shape: (B, S, E)
    V.shape: (B, S, E)
    
    returns: (B, S, E)
    """

    if attn_mask is None: attn_mask = torch.zeros(Q.size(1), K.size(1))

    energy = torch.nan_to_num( torch.softmax((torch.bmm(Q, K.mT) / (K.size(2) ** 0.5)) + attn_mask, dim=2) )

    out = torch.bmm(energy, V)
    return out

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model:int, n_heads: int):
        super().__init__()
        assert (d_model % n_heads) == 0, f"d_model ({d_model}) should be divisible by n_heads ({n_heads})"

        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.Q: Callable[[torch.Tensor], torch.Tensor] = nn.Linear(d_model, d_model)
        self.K: Callable[[torch.Tensor], torch.Tensor] = nn.Linear(d_model, d_model)
        self.V: Callable[[torch.Tensor], torch.Tensor] = nn.Linear(d_model, d_model)

        self.out: Callable[[torch.Tensor], torch.Tensor] = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None):
        """
        x.shape: (B, S, E)
        attn_mask.shape: (B, S, S) or (S, S)
        returns: (B, S, E)
        """
        B, S, E = x.shape
        # Note: E == d_model == n_heads*d_head

        q: torch.Tensor = self.Q(x).reshape(B, S, self.n_heads, self.d_head).transpose(1, 2).reshape(B*self.n_heads, S, self.d_head)
        k: torch.Tensor = self.K(x).reshape(B, S, self.n_heads, self.d_head).transpose(1, 2).reshape(B*self.n_heads, S, self.d_head)
        v: torch.Tensor = self.V(x).reshape(B, S, self.n_heads, self.d_head).transpose(1, 2).reshape(B*self.n_heads, S, self.d_head)

        if (attn_mask is not None) and (attn_mask.dim() == 3): attn_mask = attn_mask.repeat_interleave(self.n_heads, dim=0)

        out: torch.Tensor = self_attention(q, k, v, attn_mask=attn_mask).reshape(B, self.n_heads, S, self.d_head).transpose(1, 2).reshape(B, S, self.n_heads*self.d_head)

        out: torch.Tensor = self.out(out)
        return out

class LanguageModelLayer(nn.Module):
    def __init__(self, d_model:int, n_heads:int, dim_feedforward:int, dropout_p:float=0.1) -> None:
        super().__init__()

        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)

        self.feedforward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None):
        """
        x.shape: (B, S, E)
        
        returns: (B, S, E)
        """

        x = self.norm1(self.dropout(self.self_attn(x, attn_mask=attn_mask)) + x)

        x = self.norm2(self.dropout(self.feedforward(x)) + x)

        return x

class InputLayer(nn.Module):
    def __init__(self, n_mels:int, d_model:int, dropout_p: float = 0.1):
        super().__init__()

        self.conv = nn.Conv1d(n_mels, d_model, kernel_size=11, padding=5)

        self.pos_embed = PositionalEmbedding(d_model, dropout_p=dropout_p)
        self.norm = nn.LayerNorm(d_model)

        self.activation = F.gelu

    def forward(self, spectrogram: torch.Tensor):
        """
        spectrogram: torch.Tensor, shape = (B, M, S)

        returns: torch.Tensor, shape = (B, S, E)
        """

        x: torch.Tensor = self.conv(spectrogram)

        x: torch.Tensor = self.activation(self.norm(x.transpose(1, 2)))

        x: torch.Tensor = self.pos_embed(x)

        return x


class LanguageModel(nn.Module):
    def __init__(self, n_mels:int, d_model:int, n_heads:int, dim_feedforward:int, vocab_size:int, n_layers:int, max_len:int=1024, dropout_p:float=0.1) -> None:
        super().__init__()

        self.input = InputLayer(n_mels, d_model, dropout_p=dropout_p)

        self.layers: List[Callable[[torch.Tensor], torch.Tensor]] = nn.ModuleList([LanguageModelLayer(d_model, n_heads, dim_feedforward, dropout_p=dropout_p) for _ in range(n_layers)])

        self.output: Callable[[torch.Tensor], torch.Tensor] = nn.Linear(d_model, vocab_size)

        self.mask_token = -1e+25
        self.mask = torch.zeros(max_len, max_len).fill_(self.mask_token).triu_(1)

        # self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, spectrogram: torch.Tensor):
        """
        spectrogram.shape: (B, 96, S)
        
        returns: (B, S, V)
        """

        x: torch.Tensor = self.input(spectrogram)

        attn_mask = self.mask[:x.size(1), :x.size(1)].to(x.device)

        for layer in self.layers:
            x = layer(x, attn_mask = attn_mask)

        out = self.output(x)

        return out

# def test():
#     from data import Dataset, DATA_DIR
#     import os

#     dims = {
#         "n_mels":96,
#         "d_model":128, 
#         "n_heads":2,
#         "dim_feedforward":256,
#         "vocab_size":29,
#         "n_layers":2,
#         "max_len":1024, 
#         "dropout_p":0.1
#     }

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     dataset = Dataset(os.path.join(DATA_DIR, 'cv-valid-train'), os.path.join(DATA_DIR, 'cv-valid-train.csv'), n_mels=dims['n_mels'])

#     model = LanguageModel(**dims).to(device)

#     audio, text = Dataset.collate_fn([dataset[i] for i in range(32)])

#     print(f"Input.shape  : {audio.shape}")
#     print(f"Output.shape : {model(audio.to(device)).shape}")
#     print(f"Label.shape  : {text.shape}")

# if __name__ == '__main__':
#     test()