import os
import glob
import torch
import numpy as np
import librosa
from torch.utils.data import Dataset, DataLoader

class AudioDeepfakeDataset(Dataset):
    def __init__(self, processed_dir="data/processed_audio"):
        self.processed_dir = processed_dir
        self.samples = []
        
        real_dir = os.path.join(processed_dir, "real")
        fake_dir = os.path.join(processed_dir, "fake")
        
        if os.path.exists(real_dir):
            for wav_path in glob.glob(os.path.join(real_dir, "*.wav")):
                self.samples.append((wav_path, 0))
                
        if os.path.exists(fake_dir):
            for wav_path in glob.glob(os.path.join(fake_dir, "*.wav")):
                self.samples.append((wav_path, 1))

    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        wav_path, label = self.samples[idx]
        
        try:
            y, sr = librosa.load(wav_path, sr=16000, duration=5.0) # max 5 secs
            # Pad if shorter than target length
            target_length = 16000 * 5
            if len(y) < target_length:
                y = np.pad(y, (0, target_length - len(y)))
            else:
                y = y[:target_length]
                
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40) # (40, time_steps)
            mfccs = torch.from_numpy(mfccs).float()
        except:
            mfccs = torch.zeros((40, 157)).float() # fallback shape
            
        return mfccs, label

def get_audio_dataloaders(config, split_ratio=0.8):
    from torch.utils.data import random_split
    batch_size = config["model"]["batch_size"]
    
    dataset = AudioDeepfakeDataset()
    if len(dataset) == 0: return None, None
        
    train_size = int(split_ratio * len(dataset))
    test_size = len(dataset) - train_size
    generator = torch.Generator().manual_seed(42)
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=generator)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader
