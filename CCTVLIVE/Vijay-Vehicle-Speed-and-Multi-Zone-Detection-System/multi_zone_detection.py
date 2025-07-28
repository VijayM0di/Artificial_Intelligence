from ultralytics import YOLO
import cv2
import imutils
import numpy as np

# Define the colors
COLOR_GREEN = (0, 255, 0)  # Green color in BGR format
COLOR_RED = (0, 0, 255)    # Red color in BGR format
COLOR_BLUE = (255, 0, 0)   # Blue color in BGR format
COLOR_PINK = (139, 0, 255)  # Pink color in BGR format

object_zone_history = {}
# Initialize video capture
video_file_path =r"C:\Users\dazau\Downloads\U-turn-file1 (1) - Trim.mp4"
vs = cv2.VideoCapture(video_file_path)

if not vs.isOpened():
    print(f"Error: Unable to open video file {video_file_path}")
    exit(1)

# Load the YOLO model
model_yolo = YOLO("yolov9e.pt")  # Replace with your model path
print("YOLO model loaded!")

# Initialize dictionaries for tracking
initial_center_points = {}  # For storing the initial coordinates of tracked objects
turning_avg = {}            # For storing turning averages or other data
centroid_values = {}        # For storing centroid points

def derive_zones(rectangle):
    # Rectangle coordinates: (x1, y1) = top-left, (x2, y2) = bottom-right
    x1, y1 = rectangle[0]
    x2, y2 = rectangle[1]

    # Define zones based on the rectangle
    zones = {
        "red": np.array([[x1, y1], [x1, y1 + 1], [x2, y1 + 1], [x2, y1]], dtype=np.int32),  # Thin strip at the top
        "blue": np.array([[x1, y1], [x1 + 1, y1], [x1 + 1, y2], [x1, y2]], dtype=np.int32),
        # Vertical slice on the left
        "green": np.array([[x2 - 1, y1], [x2, y1], [x2, y2], [x2 - 1, y2]], dtype=np.int32),
        # Vertical slice on the right
        "pink": np.array([[x1, y2 - 1], [x2, y2 - 1], [x2, y2], [x1, y2]], dtype=np.int32),
        # Thin strip at the bottom
    }

    return zones

# Example rectangle (top-left: (90, 10), bottom-right: (650, 300))
rectangle = [(90, 10), (650, 300)]
zones = derive_zones(rectangle)

for zone, points in zones.items():
    print(f"{zone}: {points}")

def get_current_zone(center_point):
    """
    Determine the current zone based on the center point.
    """
    for zone_name, zone_points in zones.items():
        if cv2.pointPolygonTest(zone_points, center_point, False) >= 0:
            return zone_name
    return None

def check_reentry(tracking_id, current_zone):
    """
    Check if the vehicle is re-entering its entry zone after leaving it.
    """
    if tracking_id in object_zone_history:
        history = object_zone_history[tracking_id]
        if current_zone in history["visited_zones"]:
            return True  # Re-entry detected
        history["visited_zones"].add(current_zone)
    else:
        object_zone_history[tracking_id] = {"entry_zone": current_zone, "visited_zones": {current_zone}}
    return False

# Main loop to process video frames
# Dictionary to store object states
# Main loop to process video frames
while True:
    frame_exists, frame = vs.read()
    if not frame_exists:
        print("Reloading the video stream.")
        vs = cv2.VideoCapture(video_file_path)  # Reinitialize if the video ends
        continue

    frame = imutils.resize(frame, width=int(720))
    results = model_yolo.track(frame, persist=True)

    try:
        track_id_list = results[0].boxes.id.int().cpu().tolist()
    except:
        track_id_list = []

    # Clear data for disappeared objects
    for key in list(object_zone_history):
        if key not in track_id_list:
            object_zone_history.pop(key, None)

    # Process each detection result
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            try:
                tracking_id = int(box.id[0])
            except:
                continue

            # Calculate the current center point of the object
            new_center = (int((x2 + x1) / 2), int((y2 + y1) / 2))

            # Determine the current zone
            current_zone = get_current_zone(new_center)

            # Default color and label
            color = COLOR_GREEN
            label = f"ID {tracking_id}: Normal"

            if current_zone:
                # Check for re-entry
                reentered = check_reentry(tracking_id, current_zone)

                if reentered:
                    print(f"[ALERT] Object ID {tracking_id} violated rules by re-entering {current_zone}.")
                    color = COLOR_RED
                    label = f"ID {tracking_id}: Violation"
                else:
                    print(f"[INFO] Object ID {tracking_id} entered {current_zone}.")

            # Draw the bounding box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Draw all zones
    for zone_name, zone_points in zones.items():
        zone_color = COLOR_GREEN if zone_name in ["green"] else COLOR_BLUE if zone_name in ["blue"] else COLOR_PINK if zone_name in ["pink"] else (0, 0, 255)
        cv2.polylines(frame, [zone_points], isClosed=True, color=zone_color, thickness=2)

    # Display the frame
    cv2.imshow("Multi-Zone Detection with Violation Alerts", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
vs.release()
cv2.destroyAllWindows()

