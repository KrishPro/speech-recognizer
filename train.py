"""
Written by KrishPro @ KP

filename: `train.py`
"""

from data import DataModule
from model import Transformer

import torch.nn as nn
import torch.optim as optim

from pytorch_lightning import LightningModule, Trainer


class TrainModel(Transformer, LightningModule):
    def __init__(self, n_mels: int, d_model: int, n_heads: int, dim_feedforward: int, n_layers: int, vocab_size: int, dropout_p: float, learning_rate:float, max_len: int = 1024):
        super().__init__(n_mels, d_model, n_heads, dim_feedforward, n_layers, vocab_size, dropout_p, max_len)

        self.criterion = nn.CTCLoss(blank=28, zero_infinity=True)
        self.lr = learning_rate

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):
        waves, texts, input_lens, target_lens = batch       
        loss = self.criterion(self(waves).transpose(0, 1), texts, input_lens, target_lens)
        self.log("loss", loss.item())
        return loss

    def validation_step(self, batch, batch_idx):
        waves, texts, input_lens, target_lens = batch
        loss = self.criterion(self(waves).transpose(0, 1), texts, input_lens, target_lens)
        self.log("val_loss", loss.item(), prog_bar=True)
        return loss

config = {
    "trainModel": {
        "n_mels": 96,
        "d_model": 128,
        "n_heads": 2,
        "dim_feedforward": 256,
        "n_layers": 3,
        "vocab_size": 29,
        "dropout_p": 0.1,
        "max_len": 1024,
        "learning_rate": 3e-4
    },
    "data": {
        "data_dir": ".dataset",
        "batch_size": 4,
        "use_workers": True
    },
    "trainer": {
        "accelerator": "gpu",
        "max_epochs": 10,
        "devices": 1
    },
    
}


def train(config):
    model = TrainModel(**config['trainModel'])

    datamodule = DataModule(**config['data'], n_mels=config['trainModel']['n_mels'])

    trainer = Trainer(**config['trainer'])

    trainer.fit(model, datamodule)

if __name__ == '__main__':
    train(config)
