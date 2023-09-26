from concurrent.futures import ThreadPoolExecutor
from torch.nn.utils.rnn import pad_sequence
import torchaudio.functional as F
import torchaudio.transforms as T
from unidecode import unidecode
from torch.utils import data
import torch.nn as nn
import pandas as pd
import torchaudio
import librosa
import random
import string
import torch
import os

"""
Kaggle Notebook: https://www.kaggle.com/code/krishbaisoya/common-voice-3-0-data-exploration

The notebook mentioned above have examples and some speed tests for the Dataset class.
"""


class Dataset(data.Dataset):
    
    text = string.ascii_lowercase + string.punctuation + string.digits + " "

    # +1 in i, to reserve 0 for padding.
    char_to_int = {c:i+1 for i,c in enumerate(text)}
    int_to_char = {i+1:c for i,c in enumerate(text)}
    
    def __init__(self, data_dir: str, sample_rate: int, n_feats=128, freq_mask=15, time_mask=35, win_length=160, hop_length=80):
        
        self.dataframe = pd.read_csv(os.path.join(data_dir, "validated.tsv"), sep="\t")[['path', 'sentence']].dropna()
        self.clips_dir = os.path.join(data_dir, "clips")
        
        self.mel_spectrogram = T.MelSpectrogram(sample_rate, n_mels=n_feats, win_length=win_length, hop_length=hop_length)
                
        self.specaug = nn.Sequential(
            torchaudio.transforms.FrequencyMasking(freq_mask_param=freq_mask),
            torchaudio.transforms.TimeMasking(time_mask_param=time_mask)
        )
        
        self.sample_rate = sample_rate

        self.thread_executer = ThreadPoolExecutor()
                

    def __getitem__(self, idx: int):
        dpoint = self.dataframe.iloc[idx]

        fname, sentence = dpoint['path'], dpoint['sentence']
        
        audio, sr = librosa.load(os.path.join(self.clips_dir, fname), sr=self.sample_rate)
        
        audio = torch.from_numpy(audio).unsqueeze(0)

        labels = self.text_to_ints(unidecode(sentence.lower()))

        audio: torch.Tensor = self.mel_spectrogram(audio)

        audio = (audio + 1e-14).log()

        rand = random.random()

        if rand > 0.50: audio = self.specaug(audio)
        if rand > 0.75: audio = self.specaug(audio)

        error = None

        if audio.size(-1)//2 < len(labels): error = "Label length is greater than audio length"
        if audio.size(-1)//2 > 2000:        error = "Audio len is greater than 2000"
        if audio.size(0) != 1:              error = "More or Less than 1 channel"
        if len(labels) < 1:                 error = "Empty label text"

        if error:
            return self.__getitem__(idx-1 if idx!=0 else idx+1)

        return audio, labels
      
        
    def get_batch(self, start_idx: int, end_idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = list(self.thread_executer.map(self.__getitem__, range(start_idx, end_idx)))
        return self.collate_fn(batch)

    @staticmethod
    def collate_fn(batch):
        audios, labels = zip(*batch)

        audio_lengths = torch.tensor([a.size(-1)//2 for a in audios])
        audios = pad_sequence([a[0].T for a in audios]) # (T, N, E)

        label_lengths = torch.tensor([len(l) for l in labels])
        labels = pad_sequence([torch.tensor(l) for l in labels]).T # (N, S)

        return audios, labels, audio_lengths, label_lengths
    
    @classmethod
    def text_to_ints(cls, text: str):
        return [cls.char_to_int[c] for c in text]
    
    @classmethod
    def ints_to_text(cls, ints: list[int]):
        return [cls.int_to_char[i] for i in ints]


    def __len__(self):
        return len(self.dataframe)