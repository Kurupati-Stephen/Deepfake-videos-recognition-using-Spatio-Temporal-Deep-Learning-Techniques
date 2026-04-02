import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

class SpatioTemporalModel(nn.Module):
    def __init__(self, sequence_length=10, hidden_size=256):
        super(SpatioTemporalModel, self).__init__()
        self.sequence_length = sequence_length
        
        # CNN Feature Extractor
        resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        # Hook into layer4 for Grad-CAM. Output spatial map is (batch, 512, 7, 7) for 224x224 imgs.
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-2]) 
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Frame-wise Classifier
        self.frame_classifier = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1) # frame-wise fake prob
        )
        
        # Temporal LSTM
        self.lstm = nn.LSTM(input_size=512, hidden_size=hidden_size, num_layers=1, batch_first=True)
        
        # Video-wise Classifier
        self.video_classifier = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1) # video-wise fake prob
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
        
        # Sequence LSTM Analysis
        features_seq = pooled_features.view(batch_size, seq_len, -1)
        lstm_out, (h_n, c_n) = self.lstm(features_seq)
        
        video_logits = self.video_classifier(h_n[-1])
        video_probs = torch.sigmoid(video_logits).view(batch_size)
        
        return video_probs, frame_probs

if __name__ == "__main__":
    model = SpatioTemporalModel()
    dummy_input = torch.randn(2, 10, 3, 224, 224)
    vp, fp = model(dummy_input)
    print("Video Output:", vp.shape)
    print("Frame Output:", fp.shape)
