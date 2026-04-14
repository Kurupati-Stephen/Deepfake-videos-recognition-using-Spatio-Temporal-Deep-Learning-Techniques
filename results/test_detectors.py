import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from image_detector import predict_image
from audio_detector import predict_audio

print("Testing Image...")
img_result = predict_image("test.jpg")
print(img_result)

print("\nTesting Audio...")
audio_result = predict_audio("test.mp3")
print(audio_result)