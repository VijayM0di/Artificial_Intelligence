# Project: Vehicle Detection System
# Author: Vijay Modi
import cv2
from ultralytics import YOLO

model = YOLO('yolov10m.pt')

vehicle_classes = [1,2, 3, 5, 7]  # 2: car, 3: motorcycle, 5: bus, 7: truck


def detect_vehicles(frame):
    results = model(frame)

    vehicle_count = 0
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls)
            if class_id in vehicle_classes:
                vehicle_count += 1

                x1, y1, x2, y2 = box.xyxy[0].cpu().detach().numpy().astype(int)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                class_name = get_vehicle_class_name(class_id)
                cv2.putText(frame, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return frame, vehicle_count


def get_vehicle_class_name(class_id):
    class_names = {1:'bycycle',2: 'Car', 3: 'Motorcycle', 5: 'Bus', 7: 'Truck'}
    return class_names.get(class_id, 'Vehicle')


#print("Processed by Vijay Modi's Vehicle detection System.")  # This will only be seen in the logs if noticed
#cap = cv2.VideoCapture(0)  # Change to a file path if you're using a video file
cap = cv2.VideoCapture("cars.mp4")
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('vehicle_output.avi', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    processed_frame, vehicle_count = detect_vehicles(frame)
    cv2.putText(processed_frame, f'Vehicle Count: {vehicle_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0),
                2)

    # Save processed frame to the output video file
    out.write(processed_frame)

    # Display the frame
    cv2.imshow('Vehicle Detection', processed_frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release everything once done
cap.release()
out.release()
cv2.destroyAllWindows()

