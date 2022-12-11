"""
Written by KrishPro @ KP

filename: `data.py`
"""

from pytorch_lightning import LightningDataModule
from typing import Callable, List
import matplotlib.pyplot as plt
import torch.utils.data as data
import torchaudio.transforms as T
import torch.nn as nn
import torchaudio
import string
import torch
import json
import os

class Vocab:
    chars = ["'", " "] + list(string.ascii_lowercase)

    char_to_int = {c:i for i,c in enumerate(chars)}
    int_to_char = {i:c for i,c in enumerate(chars)}

    @classmethod
    def text_to_ints(cls, text:str):
        return [cls.char_to_int[char] for char in text]
    
    @classmethod
    def ints_to_text(cls, ints: List[int]):
        return ''.join([cls.int_to_char[int] for int in ints])

class Dataset(data.Dataset):
    def __init__(self, data_dir:str, split:str,  n_mels=96, freq_mask=15,time_mask=30, valid=False) -> None:
        super().__init__()

        self.data_dir = data_dir
        self.vocab = Vocab()

        self.audio_transforms: Callable[[torch.Tensor], torch.Tensor]  = T.MelSpectrogram(sample_rate=8000, win_length=160, hop_length=80, n_mels=n_mels)

        self.audio_masking: Callable[[torch.Tensor], torch.Tensor] = nn.Sequential(
            T.FrequencyMasking(freq_mask),
            T.TimeMasking(time_mask),
            T.FrequencyMasking(freq_mask),
            T.TimeMasking(time_mask),
        ) if not valid else nn.Identity()

        with open(os.path.join(data_dir, f'{split}.json')) as file:
            self.data = list(map(json.loads, file))

    def __getitem__(self, idx: int):
        filename, text = self.data[idx].values()

        wave: torch.Tensor = torchaudio.load(os.path.join(self.data_dir, filename))[0]

        wave: torch.Tensor = (self.audio_transforms(wave) + 1e-14).log()

        wave: torch.Tensor = self.audio_masking(wave)

        if wave.size(2) > 1024:
            return self.__getitem__(idx-1 if idx != 0 else idx+1)

        text = self.vocab.text_to_ints(text)

        return wave.squeeze(0).T, text

    def __len__(self):
        return len(self.data)

    @staticmethod
    def collate_fn(data):
        waves, texts = zip(*data)

        inputs_len = [wave.size(0) for wave in waves]

        targets_len = list(map(len, texts))

        waves = nn.utils.rnn.pad_sequence(waves, batch_first=True).transpose(1, 2)
        texts = nn.utils.rnn.pad_sequence(map(torch.tensor, texts), batch_first=True)

        return waves, texts, inputs_len, targets_len


class DataModule(LightningDataModule):
    def __init__(self, data_dir:str, batch_size:int, n_mels:int, use_workers:bool = True) -> None:
        super().__init__()

        self.data_dir = data_dir

        self.batch_size = batch_size

        self.n_mels = n_mels

        self.use_workers = use_workers

    def setup(self, stage:str=None):
        self.train_dataset = Dataset(self.data_dir, 'train', n_mels=self.n_mels)
        self.val_dataset = Dataset(self.data_dir, 'test', n_mels=self.n_mels)

    def train_dataloader(self):
        return data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=os.cpu_count() if self.use_workers else 0,
        persistent_workers=self.use_workers, pin_memory=True, collate_fn=Dataset.collate_fn)
    
    def val_dataloader(self):
        return data.DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=os.cpu_count() if self.use_workers else 0,
        persistent_workers=self.use_workers, pin_memory=True, collate_fn=Dataset.collate_fn)