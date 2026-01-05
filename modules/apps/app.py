import gradio as gr
import cv2
import os
import tempfile
import numpy as np
import sys
from modules.core.model_loader import load_all_models
from modules.core.detection_utils import draw_multiple_detections, MultiModelProcessor
from modules.core.image_processor import process_image
from modules.core.video_processor import VideoProcessor
from modules.core.config import CONF_THRESHOLD, MODEL_IMG_SIZE, CAMERA_WIDTH, CAMERA_HEIGHT, FRAME_SKIP

# Tải tất cả models khi khởi động
print("Đang tải tất cả models...")
all_models = load_all_models()
if not all_models or all(model is None for model in all_models.values()):
    raise RuntimeError("Không thể tải models. Vui lòng kiểm tra đường dẫn models.")

# Khởi tạo multi-model prediction processor
prediction_processor = MultiModelProcessor(all_models)

# Biến toàn cục để lưu kết quả video
last_video_output = None


def process_image_ui(image_path, auto_process=False):
    """
    Xử lý ảnh và trả về kết quả cho Gradio
    
    Args:
        image_path: Đường dẫn đến file ảnh hoặc numpy array
        auto_process: Tự động xử lý khi upload
    
    Returns:
        numpy array của ảnh đã được xử lý (RGB format)
    """
    if image_path is None:
        return None
    
    # Xử lý nếu là file path
    if isinstance(image_path, str):
        result = process_image(prediction_processor, image_path)
        if result is None:
            return None
        frame, display_frame = result
    else:
        # Nếu là numpy array từ Gradio
        frame = cv2.cvtColor(image_path, cv2.COLOR_RGB2BGR)
        # Predict với tất cả models
        prediction_processor.predict_frame(frame)
        all_results = prediction_processor.get_all_results()
        display_frame = draw_multiple_detections(frame.copy(), all_results)
    
    # Chuyển từ BGR sang RGB cho Gradio
    return cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)

# Xử lý video và trả về video đã được xử lý
def process_video_ui(video_path, save_video=False):
    global last_video_output
    
    if video_path is None:
        return None, "Vui lòng upload video"
    
    # Xử lý nếu là file path
    if isinstance(video_path, str):
        input_path = video_path
    else:
        # Nếu là dict từ Gradio
        input_path = video_path if isinstance(video_path, str) else video_path.name
    
    # Sử dụng VideoProcessor với PredictionProcessor
    video_processor = VideoProcessor(prediction_processor)
    
    # Mở video
    if not video_processor.open_video(input_path):
        return None, f"Không thể mở video: {input_path}"
    
    # Lấy thông tin video
    fps = video_processor.cap.get(cv2.CAP_PROP_FPS)
    width = int(video_processor.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_processor.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(video_processor.cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Chỉ tạo VideoWriter nếu cần lưu video
    output_path = None
    if save_video:
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        output_path = f"{base_name}_processed.mp4"
        
        # Dùng H.264 codec để tương thích browser
        fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264
        video_processor.video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        if not video_processor.video_writer.isOpened():
            # Fallback sang mp4v nếu avc1 không được hỗ trợ
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_processor.video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
        if not video_processor.video_writer.isOpened():
            video_processor.release()
            return None, "Không thể tạo file video output"
    else:
        # Không lưu video, chỉ tạo file tạm để hiển thị
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        output_path = os.path.join(tempfile.gettempdir(), f'{base_name}_processed.mp4')
        
        # Dùng H.264 codec để tương thích browser
        fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264
        video_processor.video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        if not video_processor.video_writer.isOpened():
            # Fallback sang mp4v nếu avc1 không được hỗ trợ
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_processor.video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"Đang xử lý video: {os.path.basename(input_path)}")
    
    # Xử lý video sử dụng logic tối ưu từ VideoProcessor
    while True:
        frame, ret = video_processor.read_frame()
        if not ret:
            break
        
        # Xử lý prediction (chỉ mỗi FRAME_SKIP frame)
        if video_processor.frame_count % FRAME_SKIP == 0:
            video_processor.prediction_processor.predict_frame(frame)
        
        # Copy frame để vẽ
        display_frame = frame.copy()
        
        # Vẽ detections từ kết quả mới nhất của tất cả models
        all_results = video_processor.prediction_processor.get_all_results()
        display_frame = draw_multiple_detections(display_frame, all_results)
        
        # Ghi frame vào video
        if video_processor.video_writer is not None:
            video_processor.video_writer.write(display_frame)
        
        video_processor.frame_count += 1
        
        # Hiển thị progress
        if video_processor.frame_count % 30 == 0:
            progress = (video_processor.frame_count / total_frames * 100) if total_frames > 0 else 0
            print(f"Đã xử lý {video_processor.frame_count}/{total_frames} frames ({progress:.1f}%)")
    
    # Đóng video writer
    if video_processor.video_writer is not None:
        video_processor.video_writer.release()
        video_processor.video_writer = None
    
    video_processor.release()
    
    last_video_output = output_path
    info_text = f"Đã xử lý xong: {video_processor.frame_count} frames"
    if save_video:
        info_text += f"\nVideo đã được lưu: {output_path}"
    else:
        info_text += f"\nVideo tạm (chỉ để hiển thị): {output_path}"
    
    print(info_text)
    return output_path, info_text

# Xử lý frame từ webcam cho live camera
def process_camera_frame(frame):
    if frame is None:
        return None
    
    # Chuyển từ RGB sang BGR cho OpenCV
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
    # Xử lý prediction với tất cả models
    prediction_processor.predict_frame(frame_bgr)
    all_results = prediction_processor.get_all_results()
    
    # Vẽ detections từ tất cả models
    display_frame = draw_multiple_detections(frame_bgr.copy(), all_results)
    
    # Chuyển lại sang RGB cho Gradio và đảm bảo uint8
    result = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
    
    # Đảm bảo là uint8
    if result.dtype != np.uint8:
        result = np.clip(result, 0, 255).astype(np.uint8)
    
    return result


def download_video():
    """Trả về đường dẫn video để download"""
    global last_video_output
    if last_video_output and os.path.exists(last_video_output):
        return last_video_output
    return None


# Tạo Gradio interface
with gr.Blocks(title="Student Behavior Detector", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # Student Behavior Detector
        
        Ứng dụng phát hiện hành vi học sinh trong lớp học sử dụng YOLO.
        
        **Các hành vi được phát hiện:**
        - **Hand-raising** (Giơ tay) - Màu đỏ
        - **Reading** (Đọc sách) - Màu xanh lá
        - **Writing** (Viết) - Màu xanh dương
        - **Talk** (Nói chuyện) - Màu cam
        - **Stand** (Đứng) - Màu hồng
        
        ---
        """
    )
    
    with gr.Tabs():
        # Tab 1: Image Inference
        with gr.Tab("Image Inference"):
            gr.Markdown("### Upload ảnh để phát hiện hành vi học sinh")
            with gr.Row():
                with gr.Column(scale=1):
                    image_input = gr.Image(
                        type="filepath",
                        label="Input Image",
                        sources=["upload", "clipboard"]
                    )
                    with gr.Row():
                        image_btn = gr.Button("Phân tích ảnh", variant="primary", scale=2)
                        auto_process_img = gr.Checkbox(
                            label="Tự động phân tích khi upload",
                            value=False,
                            scale=1
                        )
                with gr.Column(scale=1):
                    image_output = gr.Image(
                        type="numpy",
                        label="Output Image với Detection"
                    )
                    image_info = gr.Textbox(
                        label="Thông tin",
                        value="Upload ảnh và nhấn 'Phân tích ảnh' để bắt đầu",
                        interactive=False
                    )
            
            # Auto process khi upload
            def on_image_upload(image_path, auto_process):
                if auto_process and image_path:
                    return process_image_ui(image_path, auto_process), "Đã tự động phân tích ảnh"
                return None, "Upload ảnh và nhấn 'Phân tích ảnh' để bắt đầu"
            
            image_input.upload(
                fn=on_image_upload,
                inputs=[image_input, auto_process_img],
                outputs=[image_output, image_info]
            )
            
            image_btn.click(
                fn=lambda x: (process_image_ui(x, False), "Đã phân tích xong ảnh"),
                inputs=image_input,
                outputs=[image_output, image_info]
            )
        
        # Tab 2: Video Inference
        with gr.Tab("Video Inference"):
            gr.Markdown("### Upload video để phát hiện hành vi học sinh")
            with gr.Row():
                with gr.Column(scale=1):
                    video_input = gr.Video(
                        label="Input Video",
                        sources=["upload"]
                    )
                    with gr.Row():
                        save_video_check = gr.Checkbox(
                            label="Lưu video sau khi xử lý",
                            value=False
                        )
                    video_btn = gr.Button("Phân tích video", variant="primary")
                    gr.Markdown("**Lưu ý:** Video sẽ được xử lý từng frame với detection")
                with gr.Column(scale=1):
                    video_output = gr.Video(
                        label="Output Video với Detection"
                    )
                    video_info = gr.Textbox(
                        label="Thông tin xử lý",
                        value="Upload video và nhấn 'Phân tích video' để bắt đầu",
                        interactive=False,
                        lines=3
                    )
                    download_btn = gr.File(
                        label="Download Video",
                        visible=False
                    )
            
            def process_video_wrapper(video_path, save_video):
                output_path, info = process_video_ui(video_path, save_video)
                if output_path:
                    return output_path, info, gr.update(visible=True, value=output_path)
                return None, info, gr.update(visible=False)
            
            video_btn.click(
                fn=process_video_wrapper,
                inputs=[video_input, save_video_check],
                outputs=[video_output, video_info, download_btn]
            )
        
        # Tab 3: Live Camera
        with gr.Tab("Live Camera"):
            gr.Markdown("### Sử dụng camera để phát hiện hành vi real-time")
            gr.Markdown("**Lưu ý:** Live Camera sử dụng OpenCV để đảm bảo hiệu suất tốt nhất cho real-time detection.")
            
            camera_info = gr.Markdown("""
            ### Hướng dẫn sử dụng Live Camera
            
            **Nhấn nút "Khởi động Live Camera Demo" bên dưới để mở ứng dụng OpenCV với live camera.**
            
            **Tính năng:**
            - Live video real-time với detection
            - Hiệu suất cao, không lag
            - Phím tắt tiện lợi
            
            **Phím tắt trong Live Camera Demo:**
            - `q`: Thoát
            - `s`: Lưu ảnh hiện tại
            
            **Lưu ý:** 
            - Ứng dụng OpenCV sẽ mở trong một cửa sổ riêng
            - Để tắt, nhấn phím `q` trong cửa sổ OpenCV hoặc đóng cửa sổ
            """)
            
            def launch_opencv_demo():
                """Khởi động OpenCV Live Camera Demo"""
                try:
                    # Lấy đường dẫn đến thư mục gốc của project
                    current_file = os.path.abspath(__file__)    # modules/apps/app.py
                    apps_dir = os.path.dirname(current_file)    # modules/apps/
                    modules_dir = os.path.dirname(apps_dir)     # modules/
                    root_dir = os.path.dirname(modules_dir)     # root
                    
                    # Đường dẫn đến file run_live_camera.py
                    wrapper_file = os.path.join(root_dir, "run_live_camera.py")
                    
                    if not os.path.exists(wrapper_file):
                        return f"Không tìm thấy file run_live_camera.py tại: {wrapper_file}\nĐảm bảo file tồn tại trong thư mục gốc của project."
                    
                    # Lấy Python executable
                    python_exe = sys.executable
                    
                    # Dùng lệnh start để mở cửa sổ console mới trên Windows
                    cmd_str = f'start "Live Camera Demo" /D "{root_dir}" "{python_exe}" "{wrapper_file}"'
                    
                    # Dùng os.system để chạy lệnh start
                    result = os.system(cmd_str)
                    
                    if result == 0:
                        return "Đã khởi động Live Camera Demo!\nCửa sổ OpenCV sẽ mở trong giây lát...\nNhấn 'q' trong cửa sổ OpenCV để thoát."
                    else:
                        return f"Không thể khởi động Live Camera Demo.\nHãy chạy thủ công từ terminal:\n   python run_live_camera.py"
                            
                except Exception as e:
                    error_msg = f"Lỗi khi khởi động Live Camera Demo: {str(e)}"
                    print(error_msg)
                    import traceback
                    traceback.print_exc()
                    return f"{error_msg}\nHãy chạy thủ công: python run_live_camera.py"
            
            launch_btn = gr.Button("Khởi động Live Camera Demo", variant="primary", size="lg")
            launch_status = gr.Textbox(
                label="Trạng thái",
                value="Nhấn nút bên trên để khởi động Live Camera Demo",
                interactive=False,
                lines=3
            )
            
            launch_btn.click(
                fn=launch_opencv_demo,
                outputs=launch_status
            )

