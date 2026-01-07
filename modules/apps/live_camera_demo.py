import cv2
import os
from modules.core.config import CAMERA_WIDTH, CAMERA_HEIGHT, FRAME_SKIP_CAMERA, MODEL_IMG_SIZE
from modules.core.model_loader import load_all_models
from modules.core.detection_utils import MultiModelProcessor
from modules.core.video_processor import VideoProcessor


def main():
    print("Đang tải tất cả models...")
    all_models = load_all_models()
    if not all_models or all(model is None for model in all_models.values()):
        print("Không thể tải models. Vui lòng kiểm tra đường dẫn models.")
        return
    
    prediction_processor = MultiModelProcessor(all_models)
    save_count = 0
    video_processor = VideoProcessor(prediction_processor)
    
    print("\nĐang mở camera...")
    if not video_processor.open_camera():
        return
    
    print(f"\nFrame skip: {FRAME_SKIP_CAMERA} (chỉ xử lý mỗi {FRAME_SKIP_CAMERA} frame)")
    print(f"Model image size: {MODEL_IMG_SIZE}px")
    print("\nHướng dẫn:")
    print("   - Nhấn 'q' để thoát")
    print("   - Nhấn 's' để lưu ảnh hiện tại")
    print("   - Đang xử lý...\n")
    
    try:
        main_running = True
        
        while main_running and video_processor.cap is not None:
            frame, ret = video_processor.read_frame()
            
            if not ret:
                print("Không thể đọc frame từ camera")
                break
            
            display_frame = video_processor.process_frame(frame, False, None)
            
            if display_frame is None:
                continue
            
            window_name = 'Student Behavior Detector - Live Camera'
            cv2.imshow(window_name, display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\nĐang thoát...")
                main_running = False
                break
            elif key == ord('s'):
                save_count += 1
                filename = f'saved_frame_{save_count}.jpg'
                cv2.imwrite(filename, display_frame)
                print(f"Đã lưu ảnh: {filename}")
    
    except KeyboardInterrupt:
        print("\nNgười dùng đã dừng chương trình (Ctrl+C)")
    
    finally:
        prediction_processor.stop()
        video_processor.release()
        cv2.destroyAllWindows()
        print("Đã đóng camera và cửa sổ hiển thị")
        print(f"Tổng số frame đã xử lý: {video_processor.frame_count}")
        if save_count > 0:
            print(f"Số ảnh đã lưu: {save_count}")


if __name__ == "__main__":
    main()
