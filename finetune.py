from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data="drone.yaml",
    epochs=20,
    imgsz=416,
    batch=4,
    lr0=1e-3,
    lrf=0.01,
    warmup_epochs=3,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    flipud=0.5,
    fliplr=0.5,
    degrees=45,
    scale=0.5,
    name="drone_finetune"
)