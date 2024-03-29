from data import Dataset
from model import Model
from tqdm import tqdm
import torch.optim as optim
import torch.nn as nn
import torch
import wandb


config = {

    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),

    "batch_size": 64,

    "epochs": 4,

    "lr": 1e-3,

    "common": {
        "n_feats": 128
    },

    "model": {
        "hidden_size": 1024,
        "num_layers": 1,
        "dropout_p": 0.1
   },

   "data": {
       "data_dir": "",
       "sample_rate": 8000,
       "freq_mask": 15,
       "time_mask": 35,
       "win_length": 160,
       "hop_length": 80
   }

}

def main(config):
    wandb.init()

    model = Model(**config['model'], **config['common']).to(config['device'])
    dataset = Dataset(**config['data'], **config['common'])


    criterion = nn.CTCLoss(blank=model.vocab_size-1, zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])

    if "checkpoint" in config.keys():
        state = torch.load(config['checkpoint'])
        model.load_state_dict(state['model_state'])
        optimizer.load_state_dict(state['optimizer_state'])

        for param_group in optimizer.param_groups:
            param_group['lr'] = max(param_group['lr'], 3e-4) # setting a minimum initial limit on lr.
            # because there is no benifit in trainning with lr of 1e-7.
        
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=10_000)

    try:
        for epoch in range(config['epochs']):
            audios, labels, audio_lengths, label_lengths = (t.to(config['device']) for t in dataset.get_batch(0, config['batch_size']))
            with tqdm(range(config['batch_size'], len(dataset), config['batch_size']), desc=f"Epoch #{epoch}") as pbar:
                for i in pbar:
                    outputs = model(audios) # (T, N, C)

                    loss: torch.Tensor = criterion(outputs, labels, audio_lengths, label_lengths)

                    audios, labels, audio_lengths, label_lengths = (t.to(config['device']) for t in dataset.get_batch(i, i+config['batch_size']))

                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                    loss = loss.item()
                    pbar.set_postfix(loss=loss)
                    wandb.log({'loss': loss, 'lr': optimizer.param_groups[0]['lr']})
                    scheduler.step(loss)
                    print(str(pbar))
    except:
        pass

    print("Saving the model")
    torch.save({"model_state": model.state_dict(), "optimizer_state": optimizer.state_dict()}, "output.bin")


if __name__ == '__main__':
    main(config)
    