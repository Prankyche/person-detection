import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture("extra/test5.mp4")

width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = cap.get(cv2.CAP_PROP_FPS)

out = cv2.VideoWriter("output.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, classes=[0], conf=0.4)
    annotated = results[0].plot()

    count = len(results[0].boxes)
    cv2.putText(annotated, f"People: {count}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    out.write(annotated)

    cv2.imshow("Detection", annotated)
    # cv2.resizeWindow("Detection", 960, 540)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    if cv2.getWindowProperty("Detection", cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
out.release()
cv2.destroyAllWindows()