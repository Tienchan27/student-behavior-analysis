import sys
import os

current_file = os.path.abspath(__file__)
root_dir = os.path.dirname(current_file)
sys.path.insert(0, root_dir)

from modules.apps.live_camera_demo import main

if __name__ == "__main__":
    main()

