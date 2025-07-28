# Project: Safety-kit detection
# Author: Vijay Modi

from ultralytics import YOLO
import cv2
import cvzone
import math

cap = cv2.VideoCapture(r"C:\Program Files\PycharmProjects\Work\Learning\YARD NVR_B1 Roadway - 1(192.168.141.210)_20241029_143845_144050_ID_0100 2.avi")  # Input video


model = YOLO("ppe.pt")


classNames = ['Hardhat', 'Mask', 'Unsafe', 'Unsafe', 'Unsafe', 'Person','Safety Vest', 'machinery', 'vehicle']

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = cap.get(cv2.CAP_PROP_FPS)

output_path = "test_case-1.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

myColor = (0, 0, 255)
while True:
    success, img = cap.read()
    if not success:
        print("Video has ended or failed, exiting...")
        break

    results = model(img, stream=True)

    for r in results:
        boxes = r.boxes
        for box in boxes:

            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1


            conf = math.ceil((box.conf[0] * 100)) / 100

            cls = int(box.cls[0])
            currentClass = classNames[cls]
            print(currentClass)

            if conf > 0.5:

                if currentClass in ['NO-Hardhat', 'NO-Safety Vest', "NO-Mask"]:
                    myColor = (0, 0, 255)
                elif currentClass in ['Hardhat', 'Safety Vest', "Mask"]:
                    myColor = (0, 255, 0)  # Green for proper PPE
                else:
                    myColor = (255, 0, 0)  # Blue for other objects

                # print("Processed by Vijay Modi's ppt2 System.")  # This will only be seen in the logs if noticed
                cvzone.putTextRect(img, f'{currentClass} {conf}',
                                   (max(0, x1), max(35, y1)), scale=1, thickness=1, colorB=myColor,
                                   colorT=(255, 255, 255), colorR=myColor, offset=5)
                cv2.rectangle(img, (x1, y1), (x2, y2), myColor, 3)


    cv2.imshow("Image", img)

    out.write(img)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
