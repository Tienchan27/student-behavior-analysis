import cv2
import os
import time
from modules.core.config import (
    CAMERA_WIDTH, CAMERA_HEIGHT, FRAME_SKIP, FRAME_SKIP_CAMERA,
    VIDEO_CODEC
)
from modules.core.detection_utils import draw_multiple_detections


class VideoProcessor:
    """Xử lý video và camera"""
    
    def __init__(self, prediction_processor):
        self.prediction_processor = prediction_processor
        self.cap = None
        self.video_writer = None
        self.frame_count = 0
        self.fps_frame_count = 0
        self.fps_start_time = time.time() 
        self.fps = 0.0 
    
    def open_camera(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        if not self.cap.isOpened():
            print("Không thể mở camera. Vui lòng kiểm tra")
            return False
        
        print("Camera đã được mở thành công!")
        print(f"Độ phân giải: {CAMERA_WIDTH}x{CAMERA_HEIGHT}")
        return True
    
    def open_video(self, file_path):
        self.cap = cv2.VideoCapture(file_path)
        if not self.cap.isOpened():
            print(f"Không thể mở video: {file_path}")
            return False
        return True
    
    def setup_video_writer(self, file_path):
        if self.cap is None:
            return False
        
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        output_path = f"{base_name}_processed.mp4"
        
        fourcc = cv2.VideoWriter_fourcc(*VIDEO_CODEC)
        self.video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        if self.video_writer.isOpened():
            print(f"Đang lưu video vào: {output_path}")
            return True
        else:
            print("Không thể tạo file video output")
            self.video_writer = None
            return False
    
    def read_frame(self):   
        if self.cap is None:
            return None, False
        
        ret, frame = self.cap.read()
        return frame, ret
    
    def process_frame(self, frame, is_video_mode=False, file_path=None):
        if frame is None:
            return None
        
        self.frame_count += 1
        self.fps_frame_count += 1
        
        display_frame = frame.copy()
        
        frame_skip = FRAME_SKIP if is_video_mode else FRAME_SKIP_CAMERA
        if self.frame_count % frame_skip == 0:
            self.prediction_processor.predict_frame(frame)
        
        all_results = self.prediction_processor.get_all_results()
        display_frame = draw_multiple_detections(display_frame, all_results)
        
        if self.fps_frame_count % 30 == 0:
            elapsed = time.time() - self.fps_start_time
            self.fps = self.fps_frame_count / elapsed if elapsed > 0 else 0
            self.fps_start_time = time.time()
            self.fps_frame_count = 0
        
        source_text = f"Video: {os.path.basename(file_path)}" if is_video_mode and file_path else "Camera"
        info_text = f"{source_text} | Frame: {self.frame_count} | FPS: {self.fps:.1f} | Press 'q' to quit"
        cv2.putText(
            display_frame,
            info_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )
        
        if is_video_mode and self.video_writer is not None:
            self.video_writer.write(display_frame)
        
        return display_frame
    
    def reset_video(self):
        if self.cap is not None:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.frame_count = 0
    
    def close_video_writer(self):
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
            print("Đã lưu xong video!")
    
    def release(self):
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
        
        if self.cap is not None:
            self.cap.release()
            self.cap = None
    
    def reset_counters(self):
        self.frame_count = 0
        self.fps_frame_count = 0
        self.fps_start_time = time.time()
        self.fps = 0.0
