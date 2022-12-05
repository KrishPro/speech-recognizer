"""
Written by KrishPro @ KP

filename: `process_data.py`
"""

from tqdm import tqdm
import torchaudio
import pandas as pd
import random
import json
import os

audio_transforms = torchaudio.transforms.Resample(48000, 8000)

def load_and_save(src, dst, sample_rate=8000):
    sound, sr = torchaudio.load(src)
    
    sound = audio_transforms(sound)

    torchaudio.save(dst, sound, sample_rate)

def main(data_dir:str, output_dir:str, sample_rate:int = 8000):
    out_dir = os.path.join(output_dir, 'clips')
    if not os.path.exists(out_dir): os.makedirs(out_dir)

    train_csv = pd.read_csv(os.path.join(data_dir, 'cv-valid-train.csv'))[['filename', 'text']]
    test_csv = pd.read_csv(os.path.join(data_dir, 'cv-valid-test.csv'))[['filename', 'text']]
    dev_csv = pd.read_csv(os.path.join(data_dir, 'cv-valid-dev.csv'))[['filename', 'text']]

    data = {'train': [], 'test': [], 'dev': []}

    i = 0
    for filename, text in tqdm(train_csv.iloc, desc="Converting train files", total=len(train_csv)):
        src = os.path.join(data_dir, 'cv-valid-train', filename)
        dst = os.path.join(out_dir, f'sample-{i}.wav')

        load_and_save(src, dst, sample_rate=sample_rate)
        
        data['train'].append({'filename': f'clips/{os.path.basename(dst)}', 'text': text})
        i += 1

    for filename, text in tqdm(test_csv.iloc, desc="Converting test files", total=len(test_csv)):
        src = os.path.join(data_dir, 'cv-valid-test', filename)
        dst = os.path.join(out_dir, f'sample-{i}.wav')

        load_and_save(src, dst, sample_rate=sample_rate)

        data['test'].append({'filename': f'clips/{os.path.basename(dst)}', 'text': text})
        i += 1

    for filename, text in tqdm(dev_csv.iloc, desc="Converting dev files", total=len(dev_csv)):
        src = os.path.join(data_dir, 'cv-valid-dev', filename)
        dst = os.path.join(out_dir, f'sample-{i}.wav')

        load_and_save(src, dst, sample_rate=sample_rate)

        data['dev'].append({'filename': f'clips/{os.path.basename(dst)}', 'text': text})
        i += 1

    for k in data: random.shuffle(data[k])

    for k in data:
        with open(os.path.join(output_dir, f'{k}.json'), 'w') as file:
            file.write("\n".join(map(json.dumps, data[k])))


if __name__ == '__main__':

    main(data_dir="raw_dataset/CommonVoice", output_dir="dataset")