import os
import glob
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class ImageDeepfakeDataset(Dataset):
    def __init__(self, processed_dir, transform=None):
        self.processed_dir = processed_dir
        self.transform = transform
        self.samples = []
        
        real_dir = os.path.join(processed_dir, "real")
        fake_dir = os.path.join(processed_dir, "fake")
        
        if os.path.exists(real_dir):
            for vid_folder in os.listdir(real_dir):
                fp = os.path.join(real_dir, vid_folder)
                if os.path.isdir(fp):
                    for img_path in glob.glob(os.path.join(fp, "*.jpg")):
                        self.samples.append((img_path, 0))
                        
        if os.path.exists(fake_dir):
            for vid_folder in os.listdir(fake_dir):
                fp = os.path.join(fake_dir, vid_folder)
                if os.path.isdir(fp):
                    for img_path in glob.glob(os.path.join(fp, "*.jpg")):
                        self.samples.append((img_path, 1))

    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = cv2.imread(img_path)
        if img is None:
            img = np.zeros((224, 224, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
        if self.transform:
            img = self.transform(img)
        else:
            img = torch.from_numpy(img.transpose((2, 0, 1))).float() / 255.0
            
        return img, label

def get_image_dataloaders(config, split_ratio=0.8):
    from torch.utils.data import random_split
    processed_dir = config["paths"]["processed_frames_dir"]
    batch_size = config["model"]["batch_size"] * 4 # can afford larger batch for spatial
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((config["model"]["image_size"], config["model"]["image_size"])),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = ImageDeepfakeDataset(processed_dir, transform=transform)
    if len(dataset) == 0: return None, None
        
    train_size = int(split_ratio * len(dataset))
    test_size = len(dataset) - train_size
    generator = torch.Generator().manual_seed(42)
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=generator)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader
