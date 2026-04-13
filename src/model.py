import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnext50_32x4d, ResNeXt50_32X4D_Weights
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

class SelfAttention(nn.Module):
    def __init__(self, hidden_size):
        super(SelfAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1)
        )
        
    def forward(self, x):
        # x is (batch, seq_len, hidden_size)
        attn_weights = F.softmax(self.attention(x), dim=1) # (batch, seq_len, 1)
        context = torch.sum(attn_weights * x, dim=1) # (batch, hidden_size)
        return context, attn_weights

class SpatioTemporalModel(nn.Module):
    def __init__(self, sequence_length=10, hidden_size=256, dropout=0.0):
        super(SpatioTemporalModel, self).__init__()
        self.sequence_length = sequence_length
        
        # CNN Feature Extractor
        resnext = resnext50_32x4d(weights=ResNeXt50_32X4D_Weights.DEFAULT)
        # Hook into layer4 for Grad-CAM. Output spatial map is (batch, 2048, 7, 7) for 224x224 imgs.
        self.feature_extractor = nn.Sequential(*list(resnext.children())[:-2]) 
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        feature_dim = 2048 # ResNeXt50 outputs 2048
        
        # Frame-wise Classifier
        self.frame_classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(256, 1) # frame-wise fake prob
        )
        
        # Temporal Bi-LSTM
        self.lstm = nn.LSTM(input_size=feature_dim, hidden_size=hidden_size, num_layers=2, batch_first=True, bidirectional=True, dropout=dropout)
        
        # Attention over Bi-LSTM outputs (hidden_size * 2 due to Bi-LSTM)
        self.attention = SelfAttention(hidden_size * 2)
        
        # Video-wise Classifier
        self.video_classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(hidden_size * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 1) # video-wise fake prob
        )
        
        # Grad-CAM storage tensors
        self.gradients = None
        self.activations = None
        
    def activations_hook(self, grad):
        self.gradients = grad

    def forward(self, x):
        # x is (batch, seq, C, H, W)
        batch_size, seq_len, C, H, W = x.size()
        x_flat = x.view(batch_size * seq_len, C, H, W)
        
        # CNN Spatial Features
        spatial_features = self.feature_extractor(x_flat) 
        
        # Register hook for Grad-CAM
        if spatial_features.requires_grad:
            spatial_features.register_hook(self.activations_hook)
        self.activations = spatial_features
        
        # Pool
        pooled_features = self.avgpool(spatial_features)
        pooled_features = pooled_features.view(batch_size * seq_len, -1)
        
        # Explainable Frame Logits
        frame_logits = self.frame_classifier(pooled_features)
        frame_probs = torch.sigmoid(frame_logits).view(batch_size, seq_len)
        
        # Sequence Bi-LSTM Analysis
        features_seq = pooled_features.view(batch_size, seq_len, -1)
        lstm_out, _ = self.lstm(features_seq) # lstm_out is (batch, seq, hidden*2)
        
        # Apply Attention
        context_vector, attn_weights = self.attention(lstm_out)
        
        video_logits = self.video_classifier(context_vector)
        video_probs = torch.sigmoid(video_logits).view(batch_size)
        
        return video_probs, frame_probs
