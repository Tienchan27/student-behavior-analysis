import sys
import os

# Thêm thư mục gốc vào path
current_file = os.path.abspath(__file__)  # run_live_camera.py
root_dir = os.path.dirname(current_file)
sys.path.insert(0, root_dir)

from modules.apps.live_camera_demo import main

if __name__ == "__main__":
    main()

