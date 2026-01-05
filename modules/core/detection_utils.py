"""
Module xử lý detection và vẽ kết quả
Hỗ trợ nhiều models song song
"""

import cv2
import threading
import queue
import numpy as np
from typing import List, Optional, Dict, Tuple
from modules.core.config import CLASS_COLOR_MAP, CONF_THRESHOLD, MODEL_IMG_SIZE, USE_THREADING, IOU_THRESHOLD, MAX_DETECTIONS


def calculate_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """Tính IoU (Intersection over Union) giữa 2 boxes"""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    # Tính diện tích intersection
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
        return 0.0
    
    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    
    # Tính diện tích union
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area
    
    if union_area == 0:
        return 0.0
    
    return inter_area / union_area

# Áp dụng NMS giữa các models để loại bỏ các nhận diện bị trùng lặp
def apply_nms_cross_models(all_detections: List[Tuple]) -> List[Tuple]:
    if not all_detections:
        return []
    
    # Sort giảm dần theo confidence 
    all_detections.sort(key=lambda x: x[2], reverse=True)
    
    # Áp dụng NMS
    keep = []
    while all_detections:
        current = all_detections.pop(0)
        keep.append(current)
        
        # Loại bỏ các detections overlap với current
        remaining = []
        for det in all_detections:
            iou = calculate_iou(current[0], det[0])
            if iou < IOU_THRESHOLD or det[1] != current[1]:
                remaining.append(det)
        all_detections = remaining
    
    return keep

# Vẽ bounding boxes và labels lên frame từ list detections đã được xử lý
def draw_detections(frame, detections: List[Tuple]):
    for box, label, confidence in detections:
        x1, y1, x2, y2 = map(int, box)
        
        color = CLASS_COLOR_MAP.get(label.lower(), (255, 255, 255))
        
        # Vẽ bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color=color, thickness=2, lineType=cv2.LINE_AA)
        
        # Vẽ label với background
        label_text = f"{label} {confidence:.2f}"
        (text_width, text_height), _ = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        
        # Background cho text
        cv2.rectangle(
            frame,
            (x1, y1 - text_height - 5),
            (x1 + text_width, y1),
            color,
            -1
        )
        
        # Text
        cv2.putText(
            frame,
            label_text,
            (x1, y1 - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )
    
    return frame

# Merge và filter kết quả từ nhiều models, áp dụng NMS để loại bỏ duplicates
def merge_and_filter_results(all_results: List) -> List[Tuple]:
    if not all_results:
        return []
    
    # Lọc bỏ các None và các kết quả không có boxes
    valid_results = [r for r in all_results if r is not None and r.boxes is not None and len(r.boxes) > 0]
    
    if not valid_results:
        return []
    
    # Collect tất cả detections từ tất cả models
    all_detections = []
    for result in valid_results:
        boxes = result.boxes.xyxy.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        
        for i in range(len(boxes)):
            box = boxes[i]
            cls = int(classes[i])
            confidence = float(confidences[i])
            label = result.names[cls]
            
            # Chỉ thêm nếu confidence đủ cao
            if confidence >= CONF_THRESHOLD:
                all_detections.append((box, label, confidence))
    
    # Loại bỏ duplicates
    filtered_detections = apply_nms_cross_models(all_detections)
    # Giới hạn số lượng detections
    if len(filtered_detections) > MAX_DETECTIONS:
        filtered_detections = filtered_detections[:MAX_DETECTIONS]
    
    return filtered_detections


# Vẽ detections lên frame sau khi merge và filter
def draw_multiple_detections(frame, all_results: List):
    # Merge và filter tất cả detections
    filtered_detections = merge_and_filter_results(all_results)
    
    if not filtered_detections:
        return frame
    return draw_detections(frame, filtered_detections)


# Class để xử lý prediction với threading (cho 1 model)
class PredictionProcessor:
    
    def __init__(self, model):
        self.model = model
        self.latest_results = None
        self.results_lock = threading.Lock()
        self.running = True
        self.pred_queue = None
        self.pred_thread = None
        
        if USE_THREADING:
            self.pred_queue = queue.Queue(maxsize=2)
            self.pred_thread = threading.Thread(target=self._prediction_thread, daemon=True)
            self.pred_thread.start()
            print("Đã khởi động prediction thread")
    
    def _prediction_thread(self):
        """Thread riêng để xử lý prediction, không block việc đọc frame"""
        while self.running:
            try:
                # Lấy frame từ queue (timeout ngắn hơn để phản hồi nhanh)
                frame = self.pred_queue.get(timeout=0.05)
                
                # Dự đoán với model (tối ưu để nhanh hơn và chính xác hơn)
                outputs = self.model.predict(
                    source=frame,
                    verbose=False,
                    conf=CONF_THRESHOLD,
                    imgsz=MODEL_IMG_SIZE,
                    half=False, 
                    device='cpu',
                    max_det=MAX_DETECTIONS,
                    iou=IOU_THRESHOLD,
                    agnostic_nms=False
                )
                
                # Lưu kết quả
                with self.results_lock:
                    self.latest_results = outputs[0]
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Lỗi trong prediction thread: {e}")
                continue
    
    def predict_frame(self, frame):
        """Thêm frame vào queue để xử lý (threading) hoặc xử lý trực tiếp"""
        if USE_THREADING:
            # Thêm frame vào queue để xử lý (bỏ qua nếu queue đầy)
            try:
                self.pred_queue.put_nowait(frame)
            except queue.Full:
                pass  # Bỏ qua frame này nếu queue đầy
        else:
            # Xử lý trực tiếp (chậm hơn)
            outputs = self.model.predict(
                source=frame,
                verbose=False,
                conf=CONF_THRESHOLD,
                imgsz=MODEL_IMG_SIZE,
                max_det=MAX_DETECTIONS,
                iou=IOU_THRESHOLD
            )
            self.latest_results = outputs[0]
    
    def get_latest_results(self):
        """Lấy kết quả prediction mới nhất"""
        if USE_THREADING:
            with self.results_lock:
                return self.latest_results
        else:
            return self.latest_results
    
    def stop(self):
        """Dừng prediction thread"""
        self.running = False


# Class để xử lý prediction với nhiều models song song
class MultiModelProcessor:
    def __init__(self, models: Dict):
        self.models = {name: model for name, model in models.items() if model is not None}
        self.processors = {}
        
        # Tạo PredictionProcessor cho mỗi model
        for name, model in self.models.items():
            self.processors[name] = PredictionProcessor(model)
        
        print(f"Đã khởi tạo MultiModelProcessor với {len(self.processors)} models")
    
    # Thêm frame vào queue của tất cả processors
    def predict_frame(self, frame):
        for processor in self.processors.values():
            processor.predict_frame(frame)
    
    # Lấy kết quả từ tất cả processors
    def get_all_results(self) -> List:
        results = []
        for processor in self.processors.values():
            result = processor.get_latest_results()
            results.append(result)
        return results

    # Dừng tất cả prediction threads
    def stop(self):
        for processor in self.processors.values():
            processor.stop()
