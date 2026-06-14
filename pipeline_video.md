# Pipeline Xử Lý Video & Camera (Video Processing Pipeline)

```mermaid
flowchart TD
    Start([Input: Video/Camera]) --> V1{Source Type?}
    V1 -->|Video| V2[Open Video<br/>cv2.VideoCapture]
    V1 -->|Camera| V3[Open Camera<br/>640x480, 30fps]
    
    V2 --> V4{Opened?}
    V3 --> V5{Opened?}
    V4 -->|No| Error1[Error]
    V5 -->|No| Error2[Error]
    V4 -->|Yes| V6[Get Properties<br/>FPS, resolution]
    V5 -->|Yes| V6
    V6 --> V7[Setup Writer<br/>Codec: avc1/mp4v]
    
    V7 --> V8[Frame Loop<br/>Initialize counters]
    V8 --> V9[Read Frame]
    V9 --> V10{Valid?}
    V10 -->|No| V11[End Stream]
    V10 -->|Yes| V12[Frame Skip<br/>Video: every 5<br/>Camera: every 10]
    
    V12 --> V13{Skip?}
    V13 -->|Yes| V14[Queue Frame<br/>maxsize=5]
    V13 -->|No| V15[Use Previous]
    
    V14 --> V16[Thread Pool<br/>3 parallel threads]
    V16 --> V17[Model 1 Thread<br/>hand-raise_read_write]
    V16 --> V18[Model 2 Thread<br/>talk]
    V16 --> V19[Model 3 Thread<br/>stand]
    
    V17 --> V20[YOLO12s Inference<br/>416x416, conf=0.5]
    V18 --> V21[YOLO12s Inference<br/>416x416, conf=0.5]
    V19 --> V22[YOLO12s Inference<br/>416x416, conf=0.5]
    
    V20 --> V23[Thread-Safe Storage<br/>Lock results]
    V21 --> V24[Thread-Safe Storage<br/>Lock results]
    V22 --> V25[Thread-Safe Storage<br/>Lock results]
    
    V15 --> V26[Get All Results<br/>Collect from threads]
    V23 --> V26
    V24 --> V26
    V25 --> V26
    
    V26 --> V27[Aggregate<br/>Extract detections]
    V27 --> V28[Filter<br/>confidence >= 0.5]
    V28 --> V29[Cross-Model NMS<br/>IoU 0.45]
    V29 --> V30[Limit<br/>top 20]
    
    V30 --> V31[Draw Boxes & Labels<br/>Color-coded]
    V31 --> V32[Calculate FPS<br/>Every 30 frames]
    V32 --> V33[Overlay Info<br/>Frame count, FPS]
    
    V33 --> V34{Save?}
    V34 -->|Yes| V35[Write Frame<br/>Video file]
    V34 -->|No| V36[Display Only]
    V35 --> V37[Progress Tracking]
    V36 --> V37
    
    V37 --> V38{Continue?}
    V38 -->|Yes| V9
    V38 -->|No| V11
    
    V11 --> V39[Cleanup<br/>Release resources]
    V39 --> End([End])
    
    style Start fill:#e1f5ff
    style End fill:#c8e6c9
    style Error1 fill:#ffcdd2
    style Error2 fill:#ffcdd2
    style V16 fill:#fff9c4
    style V29 fill:#f3e5f5
    style V31 fill:#e1bee7
```

## Cách sử dụng

Copy code Mermaid ở trên (từ `flowchart TD` đến `}`) và dán vào [mermaid.live](https://mermaid.live) để xem và chỉnh sửa.
