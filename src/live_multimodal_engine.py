import os
import cv2
import torch
import numpy as np
import threading
import sounddevice as sd
import mss
from collections import deque
from PIL import Image
from torchvision import transforms

# Internal project imports
from model import SpatioTemporalModel, AudioModel, SpatialModel

class LiveForensicEngine:
    def __init__(self, config, device=None):
        self.config = config
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load Models
        self.video_model = self._load_model(
            SpatioTemporalModel(
                sequence_length=config['model']['sequence_length'],
                hidden_size=config['model']['hidden_size']
            ),
            os.path.join('models', 'best_model.pth')
        )
        
        self.audio_model = self._load_model(
            AudioModel(),
            os.path.join('models', 'best_audio_model.pth')
        )
        
        # We use a spatial-only model for faster "Image" forensics on live frames
        self.image_model = self._load_model(
            SpatialModel(),
            os.path.join('models', 'best_image_model.pth')
        )
        
        # Buffers
        self.seq_len = config['model']['sequence_length']
        self.video_buffer = deque(maxlen=self.seq_len)
        self.img_size = config['model']['image_size']
        
        self.audio_sr = 16000
        self.audio_buffer = np.array([], dtype=np.float32)
        self.audio_lock = threading.Lock()
        
        # Results
        self.latest_video_score = 0.0
        self.latest_audio_score = 0.0
        self.latest_image_score = 0.0
        self.latest_grad_cam = None # For live overlays
        
        # Preprocessing transforms
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # State
        self.running = False
        self.audio_device_index = None
        
    def _load_model(self, model, path):
        model = model.to(self.device)
        if os.path.exists(path):
            model.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))
        model.eval()
        return model
    
    @staticmethod
    def list_audio_devices():
        try:
            devices = sd.query_devices()
            return [{"id": i, "name": d['name'], "inputs": d['max_input_channels']} 
                    for i, d in enumerate(devices) if d['max_input_channels'] > 0]
        except Exception:
            return [{"id": 0, "name": "Audio Library Error (check sounddevice)", "inputs": 1}]

    def set_audio_device(self, device_id):
        self.audio_device_index = device_id

    def start_audio_capture(self):
        self.running = True
        self.audio_thread = threading.Thread(target=self._audio_loop, daemon=True)
        self.audio_thread.start()

    def _audio_callback(self, indata, frames, time, status):
        with self.audio_lock:
            # Flatten indata if it has multiple channels
            self.audio_buffer = np.append(self.audio_buffer, indata.flatten())
            # Keep only the last 5 seconds of audio
            max_samples = self.audio_sr * 5
            if len(self.audio_buffer) > max_samples:
                self.audio_buffer = self.audio_buffer[-max_samples:]

    def _audio_loop(self):
        try:
            device = self.audio_device_index
            with sd.InputStream(samplerate=self.audio_sr, channels=1, callback=self._audio_callback, device=device):
                while self.running:
                    # Run audio inference every 2 seconds
                    if len(self.audio_buffer) >= self.audio_sr * 2:
                        self._run_audio_inference()
                    sd.sleep(2000)
        except Exception as e:
            print(f"Audio capture error: {e}")

    def _run_audio_inference(self):
        import librosa
        with self.audio_lock:
            y = self.audio_buffer.copy()
            
        if len(y) < self.audio_sr: return
        
        # Extract features (match processing in audio_detector.py)
        mfccs = librosa.feature.mfcc(y=y, sr=self.audio_sr, n_mfcc=40)
        tensor_mfcc = torch.from_numpy(mfccs).float().unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            prob = self.audio_model(tensor_mfcc)
            self.latest_audio_score = prob.item()

    def process_live_video_frame(self, frame, detection_mode='haar'):
        """
        Process a single incoming frame (from any source).
        """
        # Ensure frame is in BGR for consistent processing if it came from mss
        if len(frame.shape) == 3 and frame.shape[2] == 4: # RGBA from mss
             frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
             
        # 1. Detection & Cropping
        face_img, bbox = self._extract_face(frame, mode=detection_mode)
        if face_img is None:
            return None, None
        
        # 2. Maintain Buffer
        face_tensor = self.transform(face_img)
        self.video_buffer.append(face_tensor)
        
        # 3. Trigger Video Inference if buffer is full
        if len(self.video_buffer) == self.seq_len:
            self._run_video_inference()
            
        # 4. Trigger high-res Image inference periodically
        self._run_image_inference(face_tensor)
        
        return face_img, bbox

    def _extract_face(self, frame, mode='haar'):
        # BGR to RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]
        
        if mode == 'haar':
            # Fast detection for live mode
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            if len(faces) > 0:
                (x, y, fw, fh) = faces[0]
                # Add padding
                x1, y1 = max(0, x-20), max(0, y-20)
                x2, y2 = min(w, x+fw+20), min(h, y+fh+20)
                face_crop = rgb[y1:y2, x1:x2]
                return Image.fromarray(face_crop).resize((self.img_size, self.img_size)), (x1, y1, x2, y2)
        
        # Fallback/Default: Center crop if no detection
        return Image.fromarray(rgb).resize((self.img_size, self.img_size)), (0, 0, w, h)

    def _run_video_inference(self):
        # Convert buffer to tensor shape (1, seq, C, H, W)
        input_tensor = torch.stack(list(self.video_buffer)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            v_prob, _ = self.video_model(input_tensor)
            self.latest_video_score = v_prob.item()

    def _run_image_inference(self, face_tensor):
        # face_tensor is (C, H, W)
        input_tensor = face_tensor.unsqueeze(0).to(self.device)
        with torch.no_grad():
            i_prob = self.image_model(input_tensor)
            self.latest_image_score = i_prob.item()

    def get_results(self):
        return {
            "video_risk": self.latest_video_score,
            "audio_risk": self.latest_audio_score,
            "image_risk": self.latest_image_score,
            "overall_threat": max(self.latest_video_score, self.latest_audio_score, self.latest_image_score)
        }

    def stop(self):
        self.running = False
        if hasattr(self, 'audio_thread'):
            self.audio_thread.join(timeout=1.0)
