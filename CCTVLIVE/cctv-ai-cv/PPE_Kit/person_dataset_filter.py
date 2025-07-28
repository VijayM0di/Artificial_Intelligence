import datetime
import os
import uuid

from ultralytics import YOLO
import cv2
import cvzone
import math
model = YOLO("ppe.pt")

classNames = ['Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 'Person', 'Safety Cone',
              'Safety Vest', 'machinery', 'vehicle']
for root, dirs, files in os.walk('/media/AISTORE/IQRAFOLDER/Code/TRUCK_DETECTION/dataset'):
    for filename in files:
        if filename.endswith('.jpg'):
            img = cv2.imread(os.path.join(root, filename))
            results = model(img)
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    # Bounding Box
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    # cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
                    w, h = x2 - x1, y2 - y1
                    # cvzone.cornerRect(img, (x1, y1, w, h))

                    # Confidence
                    conf = math.ceil((box.conf[0] * 100)) / 100
                    # Class Name
                    cls = int(box.cls[0])
                    currentClass = classNames[cls]
                    print(currentClass)
                    current_time = datetime.datetime.now()
                    if current_time.hour == 4 and current_time.minute > 5:
                        exit(0)
                    current_time = current_time.strftime('%Y-%m-%d_%H-%M-%S')
                    cam_id = root.split('/')[-2]
                    if conf>0.5:
                        if currentClass == 'Person':
                            cv2.imwrite(f'/media/AISTORE/IQRAFOLDER/Code/person_dataset/person_cam_{cam_id}__{current_time}.jpg', img)