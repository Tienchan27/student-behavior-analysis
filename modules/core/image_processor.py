import cv2
import os
from modules.core.config import MAX_DISPLAY_SIZE
from modules.core.detection_utils import draw_multiple_detections


# Xử lý ảnh tĩnh
def process_image(multi_model_processor, file_path):
    frame = cv2.imread(file_path)
    
    if frame is None:
        print(f"Không thể đọc ảnh: {file_path}")
        return None
    
    # Resize ảnh nếu quá lớn
    height, width = frame.shape[:2]
    if width > MAX_DISPLAY_SIZE or height > MAX_DISPLAY_SIZE:
        scale = MAX_DISPLAY_SIZE / max(width, height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        frame = cv2.resize(frame, (new_width, new_height))
    
    # Xử lý prediction với tất cả models
    print("Đang phân tích ảnh với tất cả models...")
    multi_model_processor.predict_frame(frame)
    
    # Lấy kết quả từ tất cả models
    all_results = multi_model_processor.get_all_results()
    
    # Vẽ detections từ tất cả models
    display_frame = draw_multiple_detections(frame.copy(), all_results)
    
    return frame, display_frame


def show_image(display_frame, file_path, window_name='Student Behavior Detector - Image'):
    # Hiển thị thông tin
    info_text = f"Image: {os.path.basename(file_path)} | Press 'q' to quit, 'u' to upload new"
    cv2.putText(
        display_frame,
        info_text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2
    )
    
    cv2.imshow(window_name, display_frame)
    print("Đã hiển thị ảnh. Nhấn 'q' để thoát, 'u' để upload file mới, 's' để lưu")
