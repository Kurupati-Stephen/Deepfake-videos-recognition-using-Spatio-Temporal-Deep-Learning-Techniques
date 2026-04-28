import os
import shutil
import glob
from pathlib import Path

class DeepfakeDatasetIngestor:
    def __init__(self, raw_real_dir="data/raw/real", raw_fake_dir="data/raw/fake"):
        self.raw_real_dir = Path(raw_real_dir)
        self.raw_fake_dir = Path(raw_fake_dir)
        
        self.raw_real_dir.mkdir(parents=True, exist_ok=True)
        self.raw_fake_dir.mkdir(parents=True, exist_ok=True)

    def _link_files(self, file_paths, target_dir, prefix=""):
        """ Using symbolic links instead of copying to preserve disk space for massive datasets (300GB+) """
        count = 0
        for fp in file_paths:
            src = Path(fp).resolve()
            filename = f"{prefix}_{src.name}"
            dst = target_dir / filename
            
            if not dst.exists():
                try:
                    os.symlink(src, dst)
                    count += 1
                except OSError as e:
                    print(f"Failed to symlink {src}: {e}. Trying hard link/copy...")
                    try:
                        shutil.copy2(src, dst)
                        count += 1
                    except Exception as ex:
                        print(f"Failed fallback copy: {ex}")
        return count

    def ingest_celeb_df(self, base_path):
        """ Maps Celeb-DF-v2 dataset to unified structure. """
        base = Path(base_path)
        if not base.exists():
            print(f"Celeb-DF path not found: {base}")
            return
            
        real_paths = glob.glob(str(base / "Celeb-real" / "*.mp4")) + \
                     glob.glob(str(base / "YouTube-real" / "*.mp4"))
        fake_paths = glob.glob(str(base / "Celeb-synthesis" / "*.mp4"))
        
        print(f"Found {len(real_paths)} real and {len(fake_paths)} fake videos in Celeb-DF.")
        
        r_count = self._link_files(real_paths, self.raw_real_dir, prefix="celebdf_real")
        f_count = self._link_files(fake_paths, self.raw_fake_dir, prefix="celebdf_fake")
        
        print(f"Successfully mapped {r_count} real and {f_count} fake files from Celeb-DF.")

    def ingest_faceforensics(self, base_path):
        """ Maps FaceForensics++ to unified structure. """
        base = Path(base_path)
        if not base.exists():
            print(f"FaceForensics path not found: {base}")
            return
            
        # Recursive glob to combat structure variations (e.g., c23 vs raw folder depth)
        real_paths = glob.glob(str(base / "original_sequences" / "**" / "videos" / "*.mp4"), recursive=True)
        fake_paths = glob.glob(str(base / "manipulated_sequences" / "**" / "videos" / "*.mp4"), recursive=True)
        
        print(f"Found {len(real_paths)} real and {len(fake_paths)} fake videos in FaceForensics.")
        
        r_count = self._link_files(real_paths, self.raw_real_dir, prefix="ff_real")
        f_count = self._link_files(fake_paths, self.raw_fake_dir, prefix="ff_fake")
        
        print(f"Successfully mapped {r_count} real and {f_count} fake files from FaceForensics.")

if __name__ == "__main__":
    print("Deepfake Scale Ingestor Framework")
    print("---------------------------------")
    print("Modify this script to point to your unzipped dataset paths.")
    
    # Example usage:
    # ingestor = DeepfakeDatasetIngestor()
    # ingestor.ingest_celeb_df("/Users/kurupatigrahamstaines/Downloads/Celeb-DF-v2")
    # ingestor.ingest_faceforensics("/Users/kurupatigrahamstaines/Downloads/FaceForensics")
