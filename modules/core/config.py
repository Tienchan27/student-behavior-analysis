# Độ phân giải camera
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

# Số frame bỏ qua giữa các lần prediction 
FRAME_SKIP = 10 

# Kích thước ảnh cho model
MODEL_IMG_SIZE = 320 

# Confidence threshold 
CONF_THRESHOLD = 0.5

# IOU threshold cho NMS
IOU_THRESHOLD = 0.45

# Max detections per image
MAX_DETECTIONS = 6

# Sử dụng threading để xử lý song song
USE_THREADING = True

# Màu cho các class (dùng tên class làm key)
CLASS_COLOR_MAP = {
    # Model hand-raise_read_write
    'hand-raising': (209, 54, 40),     # Đỏ
    'reading': (37, 194, 45),          # Xanh lá
    'writing': (34, 92, 240),          # Xanh dương
    # Model talk
    'talk': (255, 165, 0),             # Cam
    # Model stand
    'stand': (255, 192, 203),           # Hồng
}

# Kích thước tối đa để hiển thị ảnh
MAX_DISPLAY_SIZE = 1200

# Video codec
VIDEO_CODEC = 'avc1'  # H.264 codec
