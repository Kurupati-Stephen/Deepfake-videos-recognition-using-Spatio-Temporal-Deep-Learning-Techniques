import os
import glob
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class DeepfakeDataset(Dataset):
    def __init__(self, processed_dir, sequence_length=10, transform=None):
        self.processed_dir = processed_dir
        self.sequence_length = sequence_length
        self.transform = transform
        self.samples = []
        
        real_dir = os.path.join(processed_dir, "real")
        fake_dir = os.path.join(processed_dir, "fake")
        
        if os.path.exists(real_dir):
            for vid_folder in os.listdir(real_dir):
                fp = os.path.join(real_dir, vid_folder)
                if os.path.isdir(fp): self.samples.append((fp, 0))
        if os.path.exists(fake_dir):
            for vid_folder in os.listdir(fake_dir):
                fp = os.path.join(fake_dir, vid_folder)
                if os.path.isdir(fp): self.samples.append((fp, 1))

    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        folder_path, label = self.samples[idx]
        frames_paths = sorted(glob.glob(os.path.join(folder_path, "*.jpg")))
        
        if len(frames_paths) > self.sequence_length:
            frames_paths = frames_paths[:self.sequence_length]
        else:
            while len(frames_paths) < self.sequence_length and len(frames_paths) > 0:
                frames_paths.append(frames_paths[-1])
                
        if len(frames_paths) == 0:
            frames = torch.zeros((self.sequence_length, 3, 224, 224))
            return frames, label

        frames = []
        for path in frames_paths:
            img = cv2.imread(path)
            if img is None:
                img = np.zeros((224, 224, 3), dtype=np.uint8)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
            if self.transform:
                img = self.transform(img)
            else:
                img = torch.from_numpy(img.transpose((2, 0, 1))).float() / 255.0
            frames.append(img)
            
        frames = torch.stack(frames)
        return frames, label

def get_dataloaders(config, split_ratio=0.8):
    from torch.utils.data import random_split
    processed_dir = config["paths"]["processed_frames_dir"]
    seq_len = config["model"]["sequence_length"]
    batch_size = config["model"]["batch_size"]
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((config["model"]["image_size"], config["model"]["image_size"])),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = DeepfakeDataset(processed_dir, sequence_length=seq_len, transform=transform)
    if len(dataset) == 0: return None, None
        
    train_size = int(split_ratio * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader
