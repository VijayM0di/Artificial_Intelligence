# Project: Vehicle Accident System
# Author: Vijay Modi
# Copyright (c) 2024 Vijay Modi. All rights reserved.


import cv2
import torch
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from transformers import DetrForObjectDetection
import numpy as np
from PIL import Image

model_name = "hilmantm/detr-traffic-accident-detection"
model = DetrForObjectDetection.from_pretrained(model_name)
model.eval()

transform = Compose([
    Resize((800, 800)),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])



def process_frame(frame):
    frame_pil = Image.fromarray(frame)
    frame_transformed = transform(frame_pil)
    frame_transformed = frame_transformed.unsqueeze(0)
    with torch.no_grad():
        outputs = model(frame_transformed)

    #print("Processed by Vijay Modi's Vehicle Accident System.")  # This will only be seen in the logs if noticed

    return outputs


# Function to draw bounding boxes on the frame
def draw_bounding_boxes(frame, outputs):
    threshold = 0.3  # Lowered threshold for detection
    scores = outputs.logits.softmax(-1)[..., 1]  # Class 1 is 'accident'
    keep = scores > threshold
    if not keep.any():
        return

    boxes = outputs.pred_boxes[keep].detach().cpu().numpy()
    for box in boxes:
        x, y, w, h = box
        x1, y1, x2, y2 = int(x - w / 2), int(y - h / 2), int(x + w / 2), int(y + h / 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)


# Function to detect accidents in a video and save the output
def detect_and_save_accidents(video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        outputs = process_frame(frame)
        draw_bounding_boxes(frame, outputs)

        out.write(frame)

        # Display the frame
        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


# Example usage
video_path = r"cr.mp4"
output_path = 'output_video5.mp4'
detect_and_save_accidents(video_path, output_path)
