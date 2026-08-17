import os
from ultralytics import YOLO
from glob import glob
from pathlib import Path
from typing import Optional, Any

def find_repo_root(start: Path | Optional | Any) -> Path:
    """Return the first directory, walking up from `start`, that contains data/pose."""
    for path in (start, *start.parents):
        if (path / "data" / "pose").is_dir():
            return path
    raise FileNotFoundError(f"Could not locate the repository root (data/pose) starting from {start}")

ROOT = find_repo_root(Path.cwd().resolve())
INPUT_DIR = ROOT / "data" / "images" / "input"
OUTPUT_DIR = ROOT / "data" / "images" / "output"

os.makedirs(OUTPUT_DIR, exist_ok=True)

image_extensions = ('*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG')
all_images = []
for ext in image_extensions:
    all_images.extend(glob(os.path.join(INPUT_DIR, ext)))

all_images.sort()

target_images = all_images[:5]

if not target_images:
    print(f"Not found any images in the folder: {INPUT_DIR}")
else:
    model = YOLO("yolov8n.pt")
    
    print(f"Start analyzing image {len(target_images)}...")
    
    for img_path in target_images:
        results = model(img_path)
        
        base_name = os.path.basename(img_path)
        save_path = os.path.join(OUTPUT_DIR, f"detected_{base_name}")
        
        for result in results:
            result.save(filename=save_path)
            
        print(f"Processed and saved: {save_path}")

print("Done analyzing 5 photos!")
