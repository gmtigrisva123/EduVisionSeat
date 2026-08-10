# EduVisionSeat

Bộ công cụ cho phát hiện và phân tích tư thế/ngồi trong ảnh và video, phục vụ nghiên cứu và ứng dụng giáo dục.

## Tổng quan
- **Mục tiêu:** Cung cấp mã nguồn, notebook và mô hình mẫu để phát hiện đối tượng (ghế, học sinh) và phân tích tư thế từ ảnh/video.
- **Thư mục chính:**
	- [data](data) — chứa dữ liệu ảnh và video đầu vào. -> bị ẩn đi do bảo mật dữ liệu học sinh và những người được thí nghiệm trong dự án - ethics sẽ được ghi trong nghiên cứu và dự án.
	- [src/detect](src/detect/__init__.py#L1) — mã phát hiện (module).
	- [src/notebook](src/notebook/pose.ipynb) — notebook minh họa (kèm mô hình nhỏ `yolov8n.pt`).
	- [src/pose](src/pose) — mã xử lý tư thế (pose estimation).

## Yêu cầu
- Python 3.8+ (hoặc tương đương)
- Thư viện phổ biến: `torch`, `opencv-python`, `numpy`, `yolov8`/`ultralytics` (tùy implementation). Nếu có `requirements.txt`, cài bằng:

```bash
pip install -r requirements.txt
```

## Cấu trúc dữ liệu
- Ảnh đầu vào: đặt trong `data/images/input/`.
- Kết quả đầu ra (hình ảnh, video, logs): lưu trong `src/notebook/data/images/output/` hoặc thư mục tương ứng.

## Hướng dẫn nhanh
1. Chuẩn bị môi trường và cài package.
2. Bổ sung dữ liệu vào [data/images/input](data/images/input).
3. Chạy mô-đun phát hiện hoặc mở notebook minh họa:

```bash
# Mở notebook (ví dụ dùng jupyter)
jupyter notebook src/notebook/pose.ipynb

# Hoặc chạy script phát hiện (tùy repo):
python -m src.detect.run --input data/images/input --output src/notebook/data/images/output
```

Lưu ý: tên script/entrypoint có thể khác; xem nội dung thư mục `src/detect` để biết chi tiết.

## Mô hình mẫu
- File mô hình mẫu có sẵn: `src/notebook/yolov8n.pt` (dự phòng). Bạn có thể thay bằng mô hình lớn hơn để cải thiện chất lượng.

## Góp ý & Phát triển
- Nếu muốn mở rộng: thêm tập dữ liệu, huấn luyện lại mô hình, hoặc tích hợp bộ lọc hậu xử lý (post-processing) cho độ chính xác cao hơn.
- Gửi issues hoặc PR nếu bạn muốn góp code hoặc báo lỗi.

## License
Xem file `LICENSE` để biết chi tiết bản quyền và điều kiện sử dụng.

---
Nếu bạn muốn, tôi có thể:
- Thêm hướng dẫn cài full `requirements.txt` từ môi trường hiện tại.
- Viết script chạy mẫu `src/detect/run.py` nếu chưa có.

