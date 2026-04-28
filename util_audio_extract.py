import os
import glob
from moviepy import VideoFileClip

def extract_audio():
    os.makedirs('data/processed_audio/real', exist_ok=True)
    os.makedirs('data/processed_audio/fake', exist_ok=True)

    for cls in ['real', 'fake']:
        vids = glob.glob(f'data/raw/{cls}/*.mp4')
        for v in vids:
            name = os.path.basename(v).replace('.mp4', '.wav')
            out_path = f'data/processed_audio/{cls}/{name}'
            if not os.path.exists(out_path):
                print(f"Extracting {out_path}...")
                try:
                    clip = VideoFileClip(v)
                    if clip.audio is not None:
                        clip.audio.write_audiofile(out_path, logger=None)
                    else:
                        import numpy as np
                        from scipy.io.wavfile import write
                        # Generate 3 seconds of dummy silence
                        sys_sr = 16000
                        y = np.zeros(sys_sr * 3, dtype=np.float32)
                        write(out_path, sys_sr, y)
                    clip.close()
                except Exception as e:
                    print(f"Failed to extract {v}: {e}")

if __name__ == "__main__":
    extract_audio()
    print("Done")
