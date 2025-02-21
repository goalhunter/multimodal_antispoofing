import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Tuple
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import os
import librosa

class RandomPadding:
    def __init__(self, out_len: int = 204800, train: bool = True):
        self.out_len = out_len
        self.train = train

    def random_pad(self, signal: torch.Tensor) -> torch.Tensor:
        # Ensure signal is 1D
        if signal.dim() > 1:
            signal = signal.squeeze()
        
        input_length = signal.shape[0]
        
        if input_length >= self.out_len:
            return signal[:self.out_len]

        if self.train:
            left = np.random.randint(0, self.out_len - input_length)
        else:
            left = int(round(0.5 * (self.out_len - input_length)))

        right = self.out_len - (left + input_length)

        pad_value_left = signal[0].float().mean().to(signal.dtype)
        pad_value_right = signal[-1].float().mean().to(signal.dtype)
        
        # Create padded tensor
        padded = torch.zeros(self.out_len, dtype=signal.dtype, device=signal.device)
        padded[:left] = pad_value_left
        padded[left:left+input_length] = signal
        padded[left+input_length:] = pad_value_right

        return padded

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.random_pad(x)
    
class AudioDataset(Dataset):
    def __init__(self, tsv_file, la_directory, file_path_col='AUDIO_FILE_NAME', system_id_col='KEY',
                 sample_rate=16000, transform=None, out_len=220500):
        # Read the TSV file
        self.data = pd.read_csv(tsv_file, sep='\t')  # Changed to tab separator

        # Extract file names
        self.file_names = self.data[file_path_col].tolist()
        
        # Create labels based on SYSTEM_ID
        self.labels = [0 if system_id == 'spoof' else 1 
                      for system_id in self.data[system_id_col]]

        # Rest of your initialization remains the same
        self.la_directory = la_directory
        self.sample_rate = sample_rate
        self.transform = transform
        self.audio_data = {}
        self.load_data()
        self.transform = transform
        self.padding_transform = RandomPadding(out_len=out_len)

    def __len__(self):
        return len(self.file_names)
    
    @staticmethod
    def _load_worker(args: Tuple[int, str, Optional[int]]) -> Tuple[int, int, np.ndarray]:
        idx, filename, sample_rate = args
        wav, sample_rate = librosa.load(filename, sr=sample_rate)
        return idx, sample_rate, wav.astype(np.float32)

    def load_data(self):
        items_to_load = [
        (idx, os.path.join(self.la_directory, fname + ".flac"), self.sample_rate)
        for idx, fname in enumerate(self.file_names)
        ]

        with Pool(processes=cpu_count()) as pool:
            for idx, sr, wav in tqdm(
                pool.imap(self._load_worker, items_to_load),
                total=len(items_to_load),
                desc="Loading audio files"
            ):
                self.audio_data[idx] = {
                    'audio': wav,
                    'sample_rate': sr,
                    'label': self.labels[idx]
                }

    def __getitem__(self, idx):
        if not (0 <= idx < len(self)):
            raise IndexError

        audio = self.audio_data[idx]['audio']
        target = self.audio_data[idx]['label']

        # Convert audio to tensor if it's not already
        if not isinstance(audio, torch.Tensor):
            audio = torch.tensor(audio, dtype=torch.float32)

        # Ensure audio is 1D
        audio = audio.squeeze()

        # Apply padding transform
        audio = self.padding_transform(audio)

        if self.transform is not None:
            audio = self.transform(audio)

        # Convert the label to a tensor
        target = torch.tensor(target, dtype=torch.long)

        return audio, target
    

# CSV file paths
train_csv = '/home/STUDENTS/pb0626/Documents/Assessment/LA-flac-subset/LA_train_subset.tsv'
test_csv = '/home/STUDENTS/pb0626/Documents/Assessment/LA-flac-subset/LA_dev_subset.tsv'

train_la = '/home/STUDENTS/pb0626/Documents/Assessment/LA-flac-subset/ASVspoof2019_LA_train_subset'
test_la = '/home/STUDENTS/pb0626/Documents/Assessment/LA-flac-subset/ASVspoof2019_LA_dev_subset'

sample_rate = 16000
batch_size = 16
out_len = 6400

def load_data_files():
    train_dataset = AudioDataset(
        tsv_file=train_csv,
        la_directory=train_la,
        sample_rate=sample_rate,
        out_len=out_len
    )

    test_dataset = AudioDataset(
        tsv_file=test_csv,
        la_directory=test_la,
        sample_rate=sample_rate,
        out_len=out_len
    )


    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    return train_dataset, test_dataset, train_loader, test_loader