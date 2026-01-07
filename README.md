# Student Behavior Analysis

Ứng dụng phát hiện hành vi học sinh trong lớp học sử dụng YOLO. Hệ thống có thể phân tích ảnh tĩnh, video và phát hiện real-time từ camera.

## Cài đặt

### Yêu cầu hệ thống

- Python 3.8 trở lên
- Webcam (cho chức năng live camera)

### Cài đặt dependencies

```bash
pip install -r requirements.txt
```

## Cách sử dụng

### 1. Gradio Web UI

Chạy ứng dụng web với giao diện Gradio:

```bash
python run_app.py
```

Truy cập tại: `http://localhost:7860`

**Các tính năng:**
- **Image Inference**: Upload ảnh để phát hiện hành vi học sinh
- **Video Inference**: Upload video để phát hiện hành vi, có tùy chọn lưu video đã xử lý
- **Live Camera**: Khởi động ứng dụng OpenCV để xem live camera với detection real-time

### 2. Live Camera Demo

Ngoài truy cập qua giao diện Gradio, người dùng có thể chạy ứng dụng OpenCV standalone cho live camera:

```bash
python run_live_camera.py
```

**Tính năng:**
- Live video real-time với detection
- Hiệu suất cao, không lag
- Phím tắt:
  - `q`: Thoát
  - `s`: Lưu ảnh hiện tại

## Models

Hệ thống sử dụng 3 models YOLO chạy song song để phát hiện các hành vi:

1. **hand-raise_read_write_model**: Phát hiện 3 hành vi
   - Hand-raising (giơ tay)
   - Reading (đọc sách)
   - Writing (viết)

2. **talk_model**: Phát hiện hành vi Talk (nói chuyện)

3. **stand_model**: Phát hiện hành vi Stand (đứng)

Models được tự động load từ:
- `modules/models/student_behavior/hand-raise_read_write_model/weights/best.pt`
- `modules/models/student_behavior/talk_model/weights/best.pt`
- `modules/models/student_behavior/stand_model/weights/best.pt`

## Cấu hình

Tất cả cấu hình nằm trong file `modules/core/config.py`:

- **Frame Skip**: 10 (chỉ xử lý mỗi 10 frame để tối ưu hiệu suất)
- **Model Image Size**: 320px
- **Confidence Threshold**: 0.5
- **IOU Threshold**: 0.45 (cho NMS)
- **Max Detections**: 6 (giới hạn số detection mỗi frame)
- **Camera Resolution**: 640x480
- **Video Codec**: avc1 (H.264) với fallback sang mp4v
- **Use Threading**: True (xử lý prediction song song)

## Cấu trúc thư mục

```
student-behavior-analysis/
├── modules/                    # Tất cả code chính
│   ├── core/                   # Core modules
│   │   ├── config.py          # Cấu hình
│   │   ├── model_loader.py    # Load models
│   │   ├── detection_utils.py # Xử lý detection, NMS, vẽ kết quả
│   │   ├── image_processor.py # Xử lý ảnh tĩnh
│   │   └── video_processor.py # Xử lý video và camera
│   ├── apps/                   # Applications
│   │   ├── app.py             # Gradio Web UI
│   │   └── live_camera_demo.py # OpenCV Live Camera Demo
│   └── models/                 # Model files
│       └── student_behavior/
│           ├── hand-raise_read_write_model/
│           ├── talk_model/
│           └── stand_model/
├── run_app.py                  # Entry point cho Gradio UI
├── run_live_camera.py          # Entry point cho Live Camera
├── requirements.txt            # Dependencies
└── README.md                   # File này
```

## Màu sắc Detection

Các hành vi được hiển thị với màu sắc khác nhau:

- **Đỏ**: Hand-raising (Giơ tay)
- **Xanh lá**: Reading (Đọc sách)
- **Xanh dương**: Writing (Viết)
- **Cam**: Talk (Nói chuyện)
- **Hồng**: Stand (Đứng)

## Cách hoạt động

1. **Multi-model Inference**: Hệ thống chạy 3 models song song, mỗi model phát hiện các hành vi khác nhau
2. **Cross-model NMS**: Áp dụng Non-Maximum Suppression giữa các models để loại bỏ detections trùng lặp
3. **Threading**: Sử dụng threading để xử lý prediction không block việc đọc frame
4. **Frame Skipping**: Chỉ xử lý mỗi 10 frame để tối ưu hiệu suất cho real-time

## Lưu ý

- Đảm bảo các model files tồn tại tại đúng đường dẫn trước khi chạy
- Với video dài, quá trình xử lý có thể mất thời gian
- Live camera yêu cầu webcam hoạt động bình thường
