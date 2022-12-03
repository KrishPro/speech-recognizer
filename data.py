"""
Written by KrishPro @ KP

filename: `data.py`
"""

from typing import List

import torch.nn as nn
import torch.utils.data as data
import torchaudio.transforms as T
import torchaudio.backend.sox_io_backend as io
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
import string
import torch
import os

DATA_DIR = "../../Datasets/CommonVoice/"

class Vocab:
    chars: List[str] = ["[PAD]", " ", "'"] + list(string.ascii_lowercase)

    char_to_int = {c:i for i, c in enumerate(chars)}
    int_to_char = {i:c for i, c in enumerate(chars)}

    @staticmethod
    def text_to_ints(text: str):
        ints = [Vocab.char_to_int[char] for char in text.lower() if char in Vocab.char_to_int]
        return ints

    @staticmethod
    def ints_to_text(ints: List[int]):
        text = ''.join([Vocab.int_to_char[int] for int in ints if int in Vocab.int_to_char])
        return text


class Lambda(nn.Module):
    def __init__(self, function, rate:float=1.0):
        super().__init__()
        self.function = function
        self.rate = rate
    def forward(self, x, ):
        if self.rate > random.random():
            return self.function(x)
        else:
            return x

class Dataset(data.Dataset):
    def __init__(self, images_dir: str, csv_path: str, sample_rate=8000, freq_mask=15, time_mask=20, n_mels=96, win_length=160, hop_length=80, valid=False) -> None:
        super().__init__()

        self.walker = pd.read_csv(csv_path)[['filename', 'text']]
        self.images_dir = images_dir
        self.sample_rate = sample_rate
        self.vocab = Vocab()

        self.spec_aug = nn.Sequential(
            T.FrequencyMasking(freq_mask_param=freq_mask),
            T.TimeMasking(time_mask_param=time_mask),
            T.FrequencyMasking(freq_mask_param=freq_mask),
            T.TimeMasking(time_mask_param=time_mask)
        ) if not valid else nn.Identity()

        self.audio_transformers = nn.Sequential(
           T.Resample(48000, sample_rate),
           T.MelSpectrogram(sample_rate=sample_rate, n_mels=n_mels, win_length=win_length, hop_length=hop_length),
           Lambda(lambda x: np.log(x + 1e-14)),
           Lambda(self.spec_aug, rate=0.5)
        )

    def __getitem__(self, idx):
        filename, text = self.walker.iloc[idx]

        audio, _ = io.load(os.path.join(self.images_dir, filename))  

        audio: torch.Tensor = self.audio_transformers(audio)

        if audio.size(2) > 1024:
            return self[idx-1] if idx != 0 else self[idx+1]   

        text = self.vocab.text_to_ints(text)

        return audio.squeeze(0).T, text

    def __len__(self):
        return len(self.walker)

    @staticmethod
    def collate_fn(data):
        audios, texts = zip(*data)

        audios = nn.utils.rnn.pad_sequence(audios, batch_first=True).transpose(1, 2)

        texts = nn.utils.rnn.pad_sequence(map(torch.tensor, texts), batch_first=True)

        return audios, texts

if __name__ == '__main__':
    dataset = Dataset(os.path.join(DATA_DIR, 'cv-valid-train'), os.path.join(DATA_DIR, 'cv-valid-train.csv'))

    for audio, text in dataset:
        audio: torch.Tensor = audio
        print(audio.shape)
        plt.imshow(audio.squeeze(0))
        plt.title(text)
        plt.show()

