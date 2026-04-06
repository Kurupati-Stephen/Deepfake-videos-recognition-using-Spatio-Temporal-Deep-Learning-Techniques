import cv2
import torch
import numpy as np
from facenet_pytorch import MTCNN
from PIL import Image

class VideoPreprocessor:
    def __init__(self, image_size=224, frame_limit=15):
        self.image_size = image_size
        self.frame_limit = frame_limit
        # Initialize MTCNN for face detection
        self.mtcnn = MTCNN(keep_all=False, select_largest=True, device='cuda' if torch.cuda.is_available() else 'cpu')

    def extract_faces_from_video(self, video_path):
        """
        Extract faces from video frames.
        Returns a list of extracted face images (PIL Images) or None if video cannot be read.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            total_frames = self.frame_limit
            
        # Sample frames evenly
        frame_indices = np.linspace(0, total_frames - 1, self.frame_limit, dtype=int)
        
        extracted_faces = []
        frame_idx = 0
        success_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_idx in frame_indices:
                # MTCNN accepts PIL Image or numpy array in RGB
                # Convert BGR to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(rgb_frame)
                
                # Detect face and crop
                face = self.mtcnn(img)
                if face is not None:
                    face_np = face.permute(1, 2, 0).cpu().numpy()
                    face_np = ((face_np + 1) * 127.5).clip(0, 255).astype(np.uint8)
                    face_img = Image.fromarray(face_np).resize((self.image_size, self.image_size))
                else:
                    # Fallback if no face detected: use full frame resized
                    face_img = img.resize((self.image_size, self.image_size))
                    
                extracted_faces.append(face_img)
                success_count += 1
                    
                if success_count >= self.frame_limit:
                    break
                    
            frame_idx += 1
            
        cap.release()
        
        # Pad if not enough faces detected
        while len(extracted_faces) < self.frame_limit and len(extracted_faces) > 0:
            extracted_faces.append(extracted_faces[-1])
            
        if not extracted_faces:
            # Absolute fallback if video was totally empty/corrupted
            return [Image.new('RGB', (self.image_size, self.image_size))] * self.frame_limit
            
        return extracted_faces

    def preprocess_faces(self, faces):
        """
        Convert list of PIL images to tensor of shape (seq_len, C, H, W).
        Normalize for ResNet.
        """
        from torchvision import transforms
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        tensors = [transform(face) for face in faces]
        return torch.stack(tensors)
