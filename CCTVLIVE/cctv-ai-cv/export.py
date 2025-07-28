from ultralytics import YOLO

# Load a model
model = YOLO("models/yolov8x.pt")  # load an official model

# Export the model
onnx_file = model.export(format="onnx", dynamic=True)
