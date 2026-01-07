import cv2
import threading
import queue
import numpy as np
from typing import List, Optional, Dict, Tuple
from modules.core.config import CLASS_COLOR_MAP, CONF_THRESHOLD, MODEL_IMG_SIZE, USE_THREADING, IOU_THRESHOLD, MAX_DETECTIONS


def calculate_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """Tính IoU giữa 2 bounding boxes"""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
        return 0.0
    
    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area
    
    if union_area == 0:
        return 0.0
    
    return inter_area / union_area


def apply_nms_cross_models(all_detections: List[Tuple]) -> List[Tuple]:
    """Áp dụng NMS để loại bỏ detections trùng lặp"""
    if not all_detections:
        return []
    
    all_detections.sort(key=lambda x: x[2], reverse=True)
    
    keep = []
    while all_detections:
        current = all_detections.pop(0)
        keep.append(current)
        
        remaining = []
        for det in all_detections:
            iou = calculate_iou(current[0], det[0])
            if iou < IOU_THRESHOLD or det[1] != current[1]:
                remaining.append(det)
        all_detections = remaining
    
    return keep


def draw_detections(frame, detections: List[Tuple]):
    """Vẽ bounding boxes và labels lên frame"""
    for box, label, confidence in detections:
        x1, y1, x2, y2 = map(int, box)
        color = CLASS_COLOR_MAP.get(label.lower(), (255, 255, 255))
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), color=color, thickness=2, lineType=cv2.LINE_AA)
        
        label_text = f"{label} {confidence:.2f}"
        (text_width, text_height), _ = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        
        cv2.rectangle(
            frame,
            (x1, y1 - text_height - 5),
            (x1 + text_width, y1),
            color,
            -1
        )
        
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


def merge_and_filter_results(all_results: List) -> List[Tuple]:
    """Merge kết quả từ nhiều models và áp dụng NMS"""
    if not all_results:
        return []
    
    valid_results = [r for r in all_results if r is not None and r.boxes is not None and len(r.boxes) > 0]
    
    if not valid_results:
        return []
    
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
            
            if confidence >= CONF_THRESHOLD:
                all_detections.append((box, label, confidence))
    
    filtered_detections = apply_nms_cross_models(all_detections)
    if len(filtered_detections) > MAX_DETECTIONS:
        filtered_detections = filtered_detections[:MAX_DETECTIONS]
    
    return filtered_detections


def draw_multiple_detections(frame, all_results: List):
    """Vẽ detections từ nhiều models lên frame"""
    filtered_detections = merge_and_filter_results(all_results)
    
    if not filtered_detections:
        return frame
    return draw_detections(frame, filtered_detections)


class PredictionProcessor:
    """Xử lý prediction cho một model với threading"""
    
    def __init__(self, model):
        self.model = model
        self.latest_results = None
        self.results_lock = threading.Lock()
        self.running = True
        self.pred_queue = None
        self.pred_thread = None
        
        if USE_THREADING:
            self.pred_queue = queue.Queue(maxsize=5)
            self.pred_thread = threading.Thread(target=self._prediction_thread, daemon=True)
            self.pred_thread.start()
            print("Đã khởi động prediction thread")
    
    def _prediction_thread(self):
        """Thread xử lý prediction"""
        while self.running:
            try:
                frame = self.pred_queue.get(timeout=0.05)
                
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
                
                with self.results_lock:
                    self.latest_results = outputs[0]
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Lỗi trong prediction thread: {e}")
                continue
    
    def predict_frame(self, frame, wait=False):
        """Thêm frame vào queue để xử lý
        
        Args:
            frame: Frame cần xử lý
            wait: Nếu True, đợi queue xử lý xong frame trước đó (timeout ngắn)
        """
        if USE_THREADING:
            if wait:
                import time
                timeout = 0.1
                start_time = time.time()
                while not self.pred_queue.empty() and (time.time() - start_time) < timeout:
                    time.sleep(0.01)
            try:
                self.pred_queue.put_nowait(frame)
            except queue.Full:
                if wait:
                    try:
                        self.pred_queue.get_nowait()
                    except queue.Empty:
                        pass
                    self.pred_queue.put_nowait(frame)
                else:
                    pass
        else:
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


class MultiModelProcessor:
    """Xử lý prediction với nhiều models song song"""
    
    def __init__(self, models: Dict):
        self.models = {name: model for name, model in models.items() if model is not None}
        self.processors = {}
        
        for name, model in self.models.items():
            self.processors[name] = PredictionProcessor(model)
        
        print(f"Đã khởi tạo MultiModelProcessor với {len(self.processors)} models")
    
    def predict_frame(self, frame, wait=False):
        """Thêm frame vào queue của tất cả processors"""
        for processor in self.processors.values():
            processor.predict_frame(frame, wait=wait)
    
    def get_all_results(self) -> List:
        """Lấy kết quả từ tất cả processors"""
        results = []
        for processor in self.processors.values():
            result = processor.get_latest_results()
            results.append(result)
        return results

    def stop(self):
        """Dừng tất cả prediction threads"""
        for processor in self.processors.values():
            processor.stop()
