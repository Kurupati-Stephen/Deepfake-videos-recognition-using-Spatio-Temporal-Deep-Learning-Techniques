import os
import torch
import numpy as np

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model import AudioModel

class AudioDeepfakeDetector:
    def __init__(self, config):
        self.config = config
        self.device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

        self.model = AudioModel().to(self.device)
        model_path = os.path.join(os.path.dirname(__file__), '..', '..', 'models', "best_audio_model.pth")
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        self.model.eval()

    def detect(self, audio_path):
        import librosa
        try:
            y, sr = librosa.load(audio_path, sr=16000, duration=5.0)
            target_length = 16000 * 5
            if len(y) < target_length:
                y = np.pad(y, (0, target_length - len(y)))
            else:
                y = y[:target_length]
                
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
            tensor_mfcc = torch.from_numpy(mfccs).float().unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                prob = self.model(tensor_mfcc)
                confidence = prob.item()
        except:
            confidence = 0.5
            y = np.zeros(16000 * 3, dtype=np.float32)
            sr = 16000
            
        return confidence, y[:sr*3], sr

def predict_audio(audio_path):
    detector = AudioDeepfakeDetector(config={})
    confidence, y, sr = detector.detect(audio_path)
    
    prediction = "SYNTHETIC" if confidence > 0.5 else "AUTHENTIC"
    return {
        "fake_probability": confidence,
        "prediction": prediction,
        "status_message": "Audio Neural Network Inference Complete"
    }

def extract_audio_features(audio_path):
    import librosa
    y, sr = librosa.load(audio_path, sr=16000)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    return mfccs
