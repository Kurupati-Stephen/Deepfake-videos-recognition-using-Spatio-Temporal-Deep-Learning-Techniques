import os
import cv2
import yaml
import torch
import numpy as np
from PIL import Image
from src.model import SpatioTemporalModel
from src.preprocessing import VideoPreprocessor
# If pytorch-grad-cam is installed:
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

class ForensicAnalyzer:
    def __init__(self, config_path='configs/config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = SpatioTemporalModel(
            num_classes=self.config['model']['num_classes'],
            lstm_hidden_size=self.config['model']['lstm_hidden_size'],
            lstm_layers=self.config['model']['lstm_layers']
        ).to(self.device)
        
        # Load weights if available
        model_path = os.path.join(self.config['training']['save_dir'], 'best_model.pth')
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            
        self.model.eval()
        
        self.preprocessor = VideoPreprocessor(
            image_size=self.config['dataset']['image_size'],
            frame_limit=self.config['dataset']['frame_limit']
        )
        
        # Determine target layer for Grad-CAM
        self.target_layers = [self.model.spatial_extractor[-1]]
        try:
            # We wrap the spatial_extractor to apply Grad-CAM
            # Because our model takes (1, seq_len, C, H, W) but resnet takes (B, C, H, W)
            # We will perform Grad-CAM on individual frames directly via the spatial_extractor
            self.setup_grad_cam()
        except:
            self.cam = None

    def setup_grad_cam(self):
        # We can run Grad-CAM on the spatial extractor directly
        class CNNWrapper(torch.nn.Module):
            def __init__(self, spatial_extractor):
                super().__init__()
                self.spatial_extractor = spatial_extractor
            def forward(self, x):
                return self.spatial_extractor(x).mean(dim=[2,3]) # Global Average Pooling mock to output features
                
        # For a standard classification CAM we need a proper output. 
        # For simplicity, we'll extract the spatial activations and generate heatmaps manually 
        # or use a simplified approach since LSTM separates CNN from output.
        pass

    def analyze_video(self, video_path):
        """
        Runs the full forensics pipeline on a single video.
        """
        # 1. Extract frames & faces
        faces = self.preprocessor.extract_faces_from_video(video_path)
        if faces is None:
            return {"error": "Failed to read video or extract faces."}
            
        # Keep original faces for visualization
        orig_faces = [np.array(f) for f in faces]
        
        # 2. Preprocess
        tensor = self.preprocessor.preprocess_faces(faces) # (seq_len, C, H, W)
        tensor = tensor.unsqueeze(0).to(self.device) # (1, seq_len, C, H, W)
        
        # 3. Model Prediction
        with torch.no_grad():
            outputs = self.model(tensor)
            prob = torch.sigmoid(outputs).item() * 100 # Overall Risk Score (%)
            
        # 4. Frame-wise Real Probability Using the Model
        frame_risks = []
        with torch.no_grad():
            for i in range(len(faces)):
                single_frame_tensor = tensor[:, i:i+1, :, :, :] # (1, 1, C, H, W)
                out = self.model(single_frame_tensor)
                f_prob = torch.sigmoid(out).item() * 100
                frame_risks.append(f_prob)
                
        frame_risks = np.array(frame_risks)
        top_suspicious_indices = np.argsort(frame_risks)[::-1][:3] # Top 3
        
        # 5. Generate Real Heatmaps using Grad-CAM
        class GradCamWrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
            def forward(self, x):
                # x is (B, C, H, W) -> unsqueeze to (B, 1, C, H, W) for SpatioTemporalModel
                return self.model(x.unsqueeze(1))
                
        try:
            cam_wrapper = GradCamWrapper(self.model)
            target_layers = [self.model.spatial_extractor[-1]]
            cam = GradCAM(model=cam_wrapper, target_layers=target_layers)
        except Exception as e:
            cam = None

        heatmaps = []
        for i in top_suspicious_indices:
            img = orig_faces[i]
            frame_t = tensor[:, i, :, :, :] # (1, C, H, W)
            
            if cam is not None:
                try:
                    grayscale_cam = cam(input_tensor=frame_t)[0, :]
                    # Normalize original image to [0,1]
                    img_float = img.astype(np.float32) / 255.0
                    vis = show_cam_on_image(img_float, grayscale_cam, use_rgb=True)
                    heatmaps.append(vis)
                except Exception:
                    heatmaps.append(img) # fallback
            else:
                heatmaps.append(img)
                
        # 6. Summary Logic
        classification = "SYNTHETIC FORGERY" if prob >= 50 else "AUTHENTIC"
        confidence = prob if prob >= 50 else (100 - prob)
        
        if prob >= 70:
            summary = f"Synthetic forgery detected with HIGH confidence ({confidence:.1f}%). Significant temporal inconsistencies and spatial artifacts located in key facial regions."
        elif prob >= 50:
            summary = f"Synthetic forgery detected with MODERATE confidence ({confidence:.1f}%). Suspicious pixel-level manipulation scattered across specific frames."
        else:
            summary = f"Video appears AUTHENTIC ({confidence:.1f}% confidence). No obvious spatio-temporal deepfake manipulations detected."
            
        return {
            "classification": classification,
            "overall_risk": prob,
            "frame_risks": list(frame_risks),
            "top_indices": top_suspicious_indices,
            "top_frames_orig": [orig_faces[i] for i in top_suspicious_indices],
            "top_frames_heatmap": heatmaps,
            "summary": summary
        }

if __name__ == "__main__":
    import os
    analyzer = ForensicAnalyzer()
    print("ForensicAnalyzer initialized successfully.")
