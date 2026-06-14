# Pipeline Xử Lý Ảnh Tĩnh (Image Processing Pipeline)

```mermaid
flowchart TD
    Start([Input Image]) --> I1[Load Image<br/>cv2.imread]
    I1 --> I2{Valid?}
    I2 -->|No| Error1[Return Error]
    I2 -->|Yes| I3[Preprocess<br/>Resize if > 1200px<br/>Keep BGR format]
    
    I3 --> I4[Multi-Model Inference<br/>3 YOLO12s parallel<br/>predict_frame]
    I5 --> I6[Model 1<br/>hand-raise_read_write<br/>416x416, conf=0.5]
    I5 --> I7[Model 2<br/>talk<br/>416x416, conf=0.5]
    I5 --> I8[Model 3<br/>stand<br/>416x416, conf=0.5]
    
    I6 --> I9[Aggregate Results<br/>boxes, classes, confidences]
    I7 --> I9
    I8 --> I9
    
    I9 --> I10[Filter<br/>confidence >= 0.5]
    I10 --> I11[Cross-Model NMS<br/>IoU threshold 0.45]
    I11 --> I12[Limit Detections<br/>top 20]
    
    I12 --> I13[Draw Boxes & Labels<br/>Color-coded by class]
    I13 --> I14[Output Image<br/>Return to UI]
    I14 --> End([Display])
    
    style Start fill:#e1f5ff
    style End fill:#c8e6c9
    style Error1 fill:#ffcdd2
    style I5 fill:#fff9c4
    style I11 fill:#f3e5f5
    style I13 fill:#e1bee7
```

## Cách sử dụng

Copy code Mermaid ở trên (từ `flowchart TD` đến `}`) và dán vào [mermaid.live](https://mermaid.live) để xem và chỉnh sửa.
