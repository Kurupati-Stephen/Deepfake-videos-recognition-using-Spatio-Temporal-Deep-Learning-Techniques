import os
import glob
import yaml
import cv2
import numpy as np

def load_config(config_path="configs/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def generate_dummy_video(output_path, num_frames=30, label="real"):
    """Generates a dummy .mp4 video for testing the pipeline."""
    height, width = 480, 640
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 10.0, (width, height))
    
    # Generate simple frames (mostly green for real, red for fake)
    for i in range(num_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        if label == "fake":
            # Red-ish background indicating fake
            frame[:] = (0, 0, 150 + (i % 105))
        else:
            # Green-ish background indicating real
            frame[:] = (0, 150 + (i % 105), 0)
            
        # Draw a face-like circle to help with OpenCV Haar cascade fallback
        cv2.circle(frame, (width//2, height//2), 100, (255, 220, 190), -1) # face base
        cv2.circle(frame, (width//2 - 35, height//2 - 20), 10, (0, 0, 0), -1) # left eye
        cv2.circle(frame, (width//2 + 35, height//2 - 20), 10, (0, 0, 0), -1) # right eye
        
        # Add some text
        cv2.putText(frame, f"Label: {label}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f"Frame: {i}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        out.write(frame)
        
    out.release()
    print(f"Generated dummy video at {output_path}")

def inspect_and_prepare_dataset():
    config = load_config()
    desktop_dataset_path = config["paths"]["desktop_dataset"]
    
    print(f"Inspecting path: {desktop_dataset_path}")
    if not os.path.exists(desktop_dataset_path):
        print(f"WARNING: The directory {desktop_dataset_path} does not exist.")
    else:
        print(f"Found dataset directory: {desktop_dataset_path}")
        
    # Search for videos in the desktop dataset
    desktop_videos = []
    for ext in ["*.mp4", "*.avi"]:
        desktop_videos.extend(glob.glob(os.path.join(desktop_dataset_path, "**", ext), recursive=True))
        
    print(f"Found {len(desktop_videos)} video files in the Desktop dataset.")
    
    # Prepare local workspace dataset directories
    raw_real = config["paths"]["raw_real_dir"]
    raw_fake = config["paths"]["raw_fake_dir"]
    os.makedirs(raw_real, exist_ok=True)
    os.makedirs(raw_fake, exist_ok=True)
    
    if len(desktop_videos) == 0:
        print("\n[INFO] The Desktop dataset is incomplete (missing video files).")
        print("[INFO] Creating a dummy dataset to allow the pipeline to function.")
        
        # Check if dummy data already exists
        real_vids = glob.glob(os.path.join(raw_real, "*.mp4"))
        fake_vids = glob.glob(os.path.join(raw_fake, "*.mp4"))
        
        if len(real_vids) == 0 and len(fake_vids) == 0:
            print("Generating 5 real and 5 fake dummy videos...")
            for i in range(5):
                generate_dummy_video(os.path.join(raw_real, f"dummy_real_{i}.mp4"), num_frames=30, label="real")
                generate_dummy_video(os.path.join(raw_fake, f"dummy_fake_{i}.mp4"), num_frames=30, label="fake")
        else:
            print("Dummy dataset already exists.")
    else:
        # User implies dataset has videos in future, we would copy/symlink them.
        print("\n[INFO] Real videos found. You should copy or symlink them into data/raw/real and data/raw/fake.")

if __name__ == "__main__":
    inspect_and_prepare_dataset()
