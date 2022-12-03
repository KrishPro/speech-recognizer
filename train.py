"""
Written by KrishPro @ KP

filename: `train.py`
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from data import DataModule
from model import LanguageModel
from typing import Tuple
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping



class Model(LightningModule):
    def __init__(self, n_mels:int, d_model:int, n_heads:int, dim_feedforward:int, vocab_size:int, n_layers:int, lr:float, log_interval:int=None, max_len:int=1024, dropout_p:float=0.1) -> None:
        super().__init__()

        self.model = LanguageModel(n_mels=n_mels, d_model=d_model, n_heads=n_heads, dim_feedforward=dim_feedforward, vocab_size=vocab_size, n_layers=n_layers, max_len=max_len, dropout_p=dropout_p)

        self.criterion = nn.CTCLoss(blank=28, zero_infinity=True)
        
        self.log_interval = log_interval
        self.lr = lr

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.model.parameters(), self.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                                        optimizer, mode='min',
                                        factor=0.50, patience=6)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx):
        audio, target, input_lens, target_lens = batch

        input: torch.Tensor = F.log_softmax(self.model(audio), dim=-1)

        loss: torch.Tensor = self.criterion(input.transpose(0, 1), target, input_lens, target_lens)
        
        if (self.log_interval is not None) and (batch_idx % self.log_interval == 0): print(f"#{self.current_epoch} | #{batch_idx} | {loss.detach()}")

        self.log("train_loss", loss)
        self.log("lr", self.optimizers().param_groups[0]['lr'], prog_bar=True)

        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx):
        audio, target, input_lens, target_lens = batch

        input: torch.Tensor = F.log_softmax(self.model(audio), dim=-1)

        loss: torch.Tensor = self.criterion(input.transpose(0, 1), target, input_lens, target_lens)
        
        self.log("val_loss", loss, prog_bar=True)

        return loss

config = {
    "dims": {
        "n_mels":96,
        "d_model":128, 
        "n_heads":2,
        "dim_feedforward":256,
        "vocab_size":29,
        "n_layers":2,
        "max_len":1024, 
        "dropout_p":0.1
    },
    "data_dir": "../../Datasets/CommonVoice/",
    "batch_size": 4,
    "use_workers": True,
    "log_interval": None,
    "learning_rate": 1e-3,
    "trainer": {
        "accelerator": "gpu",
        "devices": 1, 
        "precision": 16,
        "accumulate_grad_batches": 1,
        # "overfit_batches": 2,
        "max_epochs": 100
    },
    "early_stop": {
        "monitor": 'val_loss',
        "min_delta": 0, 
        "patience": 3,
        "verbose":False,
        "mode":'min',
        "strict":True
    }
}

def train(config):

    datamodule = DataModule(config['data_dir'], batch_size=config['batch_size'], use_workers=config['use_workers'], n_mels=config['dims']['n_mels'])
    
    model = Model(**config['dims'], lr=config['learning_rate'], log_interval=config['log_interval'])

    callbacks = [EarlyStopping(**config['early_stop'])] if config['early_stop'] is not False else []

    trainer = Trainer(**config['trainer'], callbacks=callbacks)

    trainer.fit(model, datamodule)

if __name__ == '__main__':
    train(config)