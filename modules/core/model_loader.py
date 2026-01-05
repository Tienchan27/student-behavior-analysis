import os
from typing import Dict, Optional, Any
from ultralytics import YOLO

# Đường dẫn các models
MODEL_PATHS = {
    'hand_raise_read_write': 'modules/models/student_behavior/hand-raise_read_write_model/weights/best.pt',
    'talk': 'modules/models/student_behavior/talk_model/runs/detect/train/weights/best.pt',
    'stand': 'modules/models/student_behavior/stand_model/runs/detect/train/weights/best.pt'
}


def load_all_models() -> Dict[str, Any]:
    models = {}
    
    print("Đang tải tất cả models...")
    for model_name, model_path in MODEL_PATHS.items():
        print(f"\n---- Đang tải {model_name} model...")
        if not os.path.exists(model_path):
            print(f"Không tìm thấy {model_name} model tại: {model_path}")
            models[model_name] = None
            continue
        
        try:
            model = YOLO(model_path)
            models[model_name] = model
            print(f"{model_name} model đã được tải thành công!")
        except Exception as e:
            print(f"Lỗi khi tải {model_name} model: {e}")
            models[model_name] = None
    
    # Kiểm tra xem có ít nhất 1 model được load không
    loaded_models = [name for name, model in models.items() if model is not None]
    if not loaded_models:
        print("\nKhông có model nào được load thành công!")
        return models
    
    print(f"\nĐã load thành công {len(loaded_models)}/{len(MODEL_PATHS)} models: {', '.join(loaded_models)}")
    return models
