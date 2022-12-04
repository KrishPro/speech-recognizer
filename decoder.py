"""
Written by KrishPro @ KP

filename: `decoder.py`
"""

from data import Vocab
from model import LanguageModel
import torchaudio.transforms as T
import torchaudio.backend.sox_io_backend as io
import torch.nn as nn
import torch
import os

class Audio:
    def __init__(self, sample_rate=8000, freq_mask=15, time_mask=20, n_mels=96, win_length=160, hop_length=80,) -> None:
        self.audio_transforms = nn.Sequential(
           T.Resample(48000, sample_rate),
           T.MelSpectrogram(sample_rate=sample_rate, n_mels=n_mels, win_length=win_length, hop_length=hop_length)
        )

    def load_audio(self, audio_path):
        wavform, _ = io.load(audio_path)

        audio: torch.Tensor = self.audio_transforms(wavform)

        audio: torch.Tensor = torch.log(audio + 1e-14)

        return audio

def process_checkpoint(input_path: str, output_path: str):
    ckpt: dict = torch.load(input_path)

    dims = {
        "n_mels":96,
        "d_model":128, 
        "n_heads":2,
        "dim_feedforward":512,
        "vocab_size":29,
        "n_layers":2,
        "max_len":1024, 
        "dropout_p":0.1
    }

    state_dict = {k[6:]:v for k, v in ckpt['state_dict'].items()}

    ckpt: dict = {'dims': dims, 'state_dict': state_dict}

    if (os.path.dirname(output_path) != '') and (not os.path.exists(os.path.dirname(output_path))): os.makedirs(os.path.dirname(output_path))

    torch.save(ckpt, output_path)


def main():
    model:LanguageModel =  LanguageModel.load_from_ckpt('Output/model.ckpt').eval()

    audio_manager = Audio()

    audio = audio_manager.load_audio('sample-000001.mp3')

    output: torch.Tensor = model(audio)

    output = output.argmax(2)[0].tolist()

    output = [o for i, o in enumerate(output) if o != output[i-1] and o != 28]

    output = Vocab.ints_to_text(output)

    print(output)



if __name__ == '__main__':
    main()