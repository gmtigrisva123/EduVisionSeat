import os
from ultralytics import YOLO
from glob import glob

INPUT_DIR = "data/images/input"
OUTPUT_DIR = "data/images/output"

os.makedirs(OUTPUT_DIR, exist_ok=True)

image_extensions = ('*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG')
all_images = []
for ext in image_extensions:
    all_images.extend(glob(os.path.join(INPUT_DIR, ext)))

all_images.sort()

# Lấy 5 ảnh
target_images = all_images[:5]

if not target_images:
    print(f"Not found any images in the folder: {INPUT_DIR}")
else:
    model = YOLO("yolov8n.pt")
    
    print(f"Start analyzing image {len(target_images)}...")
    
    for img_path in target_images:
        # Chạy model
        results = model(img_path)
        
        # Lấy tên file gốc
        base_name = os.path.basename(img_path)
        # Tạo đường dẫn lưu mới
        save_path = os.path.join(OUTPUT_DIR, f"detected_{base_name}")
        
        # Lưu ảnh đã vẽ bounding box vào thư mục đích
        for result in results:
            result.save(filename=save_path)
            
        print(f" Đã xử lý và lưu: {save_path}")

print(" Hoàn thành phân tích 5 ảnh!")
