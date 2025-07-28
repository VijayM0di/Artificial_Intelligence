from ultralytics import YOLO
model = YOLO('yolov8x.pt')

model.train(data='/media/AISTORE/IQRAFOLDER/Code/PPE-Kit-demo/PPE-Dataset-v3i/data.yaml', epochs=100, device=0)
