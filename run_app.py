"""
File để chạy Gradio Web UI
Chạy từ thư mục gốc: python run_app.py
"""

import sys
import os

# Thêm thư mục gốc vào path
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.insert(0, root_dir)

from modules.apps.app import demo

if __name__ == "__main__":
    print("Đang khởi động Gradio Web UI...")
    print("Truy cập tại: http://localhost:7860")
    demo.queue().launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )

