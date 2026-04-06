import os
import glob
import torch
from torch.utils.data import Dataset, DataLoader
from src.preprocessing import VideoPreprocessor

class DeepfakeVideoDataset(Dataset):
    def __init__(self, data_dir, config, split='train'):
        self.data_dir = data_dir
        self.real_subfolder = config['dataset']['real_subfolder']
        self.fake_subfolder = config['dataset']['fake_subfolder']
        self.frame_limit = config['dataset']['frame_limit']
        self.image_size = config['dataset']['image_size']
        self.preprocessor = VideoPreprocessor(image_size=self.image_size, frame_limit=self.frame_limit)
        
        self.samples = [] # list of (video_path, label) where label=0 (Authentic), 1 (Synthetic)
        self._load_metadata(split)
        
    def _load_metadata(self, split):
        real_dir = os.path.join(self.data_dir, self.real_subfolder)
        fake_dir = os.path.join(self.data_dir, self.fake_subfolder)
        
        real_vids = glob.glob(os.path.join(real_dir, '*.mp4')) + glob.glob(os.path.join(real_dir, '*.mov'))
        fake_vids = glob.glob(os.path.join(fake_dir, '*.mp4')) + glob.glob(os.path.join(fake_dir, '*.mov'))
        
        all_samples = [(vid, 0) for vid in real_vids] + [(vid, 1) for vid in fake_vids]
        
        # Simple deterministic split
        all_samples.sort() 
        import random
        random.seed(42)
        random.shuffle(all_samples)
        
        n_samples = len(all_samples)
        train_end = int(0.8 * n_samples)
        val_end = int(0.9 * n_samples)
        
        if split == 'train':
            self.samples = all_samples[:train_end]
        elif split == 'val':
            self.samples = all_samples[train_end:val_end]
        else:
            self.samples = all_samples[val_end:]
            
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        video_path, label = self.samples[idx]
        faces = self.preprocessor.extract_faces_from_video(video_path)
        
        if faces is None:
            # Fallback if detection fails
            faces = [self.preprocessor.Image.new('RGB', (self.image_size, self.image_size))] * self.frame_limit
            
        x = self.preprocessor.preprocess_faces(faces)
        
        return x, torch.tensor(label, dtype=torch.float32)

def get_dataloader(data_dir, config, split='train', batch_size=8, shuffle=True):
    dataset = DeepfakeVideoDataset(data_dir, config, split)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)
