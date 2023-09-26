from data import Dataset
import torch.nn.functional as F
import torch.nn as nn
import torch


class Model(nn.Module):
    
    vocab_size = len(Dataset.text) + 1 # [0=padding, 1...len=letters, len+1=blank(CTC)]
    
    def __init__(self, n_feats: int, hidden_size: int, num_layers=1, dropout_p=0.1):
        super().__init__()


        self.conv = nn.Conv1d(n_feats, n_feats, kernel_size=10, stride=2, padding=10//2)

        self.norm1 = nn.Sequential(
            nn.LayerNorm(n_feats),
            nn.ReLU(),
            nn.Dropout(dropout_p)
        )

        self.feed = nn.Sequential(
            nn.Linear(n_feats, n_feats),
            nn.LayerNorm(n_feats),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(n_feats, n_feats),
            nn.LayerNorm(n_feats),
            nn.ReLU(),
            nn.Dropout(dropout_p)
        )

        self.lstm = nn.LSTM(n_feats, hidden_size, num_layers=num_layers)
        
        self.norm2 = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_p)
        )

        self.pred = nn.Linear(hidden_size, self.vocab_size)


    def forward(self, x: torch.Tensor): # (L, N, C)
        
        # Doing permutes because the conv layer have special requirements.

        x = x.permute(1, 2, 0) # (N, C, L)
        x = self.conv(x)       # (N, C, L)
        x = x.permute(2, 0, 1) # (L, N, C)

        # Note: the `L` after conv is `L//2`.

        x = self.norm1(x)      # (L, N, C)

        x = self.feed(x)       # (L, N, C)

        x = self.lstm(x)[0]    # (L, N, H)

        x = self.norm2(x)      # (L, N, H)

        x = self.pred(x)       # (L, N, V)

        return F.log_softmax(x, dim=2)
        
        


if __name__ == '__main__':

    T, B, E, H, V = (100, 8, 128, 1024, 70)

    model = Model(E, H)

    sample_audio_batch = torch.randn(T, B, E) # (time, batch_size, encodings)

    output = model(sample_audio_batch)

    print(f"Current: {tuple(output.shape)}")
    print(f"Final:   ({(T//2)+1}, {B}, {V})")