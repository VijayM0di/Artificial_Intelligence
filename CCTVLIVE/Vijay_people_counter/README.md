# People Counter

## Overview
The People Counter project utilizes YOLOv8 (You Only Look Once) for real-time object detection and tracking. It specifically counts the number of people crossing designated boundaries in a video feed or a pre-recorded video. This tool is useful for applications such as monitoring foot traffic in public areas, retail stores, or events.

## Features
- **Real-Time Detection**: Detects people in a video feed using the YOLOv8 model.
- **Customizable Boundaries**: Tracks people crossing user-defined boundaries (e.g., entry and exit points).
- **Accurate Tracking**: Uses the SORT (Simple Online and Realtime Tracking) algorithm for consistent ID assignment.
- **Visual Overlays**: Displays bounding boxes, IDs, and crossing counts directly on the video feed.

## Requirements

### Hardware
- A computer with a powerful GPU (recommended: NVIDIA RTX series).

### Software
- Python 3.8 or later
- Required Python libraries:
  - `numpy`
  - `opencv-python`
  - `opencv-python-headless`
  - `cvzone`
  - `ultralytics`
  - `sort`

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Quantmbot-AI/vijay_people_counter.git
   cd vijay_people_counter
   ```

2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the YOLOv8 model weights and place them in the project directory. You can download it from [Ultralytics](https://github.com/ultralytics/ultralytics).

4. Prepare your video file or ensure your webcam is accessible.

## File Structure
```
people-counter/
|-- ppl_counter.py          # Main program file
|-- requirements.txt        # Dependencies
|-- README.md               # Project description and instructions
```

## Usage

1. Modify the `cap` variable in `main.py` to use either a webcam or a video file:
   ```python
   # For webcam input
   cap = cv2.VideoCapture(0)

   # For video file input
   cap = cv2.VideoCapture("path_to_your_video.mp4")
   ```

2. Run the program:
   ```bash
   python main.py
   ```

3. Observe the video feed with overlays for detected people, IDs, and counts. Counts for upward and downward crossings are displayed on the screen.

## Customization

- **Boundaries**:
  Update the `limitsUp` and `limitsDown` variables to define custom boundary lines for counting.
  ```python
  limitsUp = [103, 161, 296, 161]
  limitsDown = [527, 489, 735, 489]
  ```

- **Confidence Threshold**:
  Adjust the confidence threshold for detections in `main.py`:
  ```python
  if currentClass == "person" and conf > 0.3:
  ```

## Known Issues
- The performance may degrade on systems without a GPU.
- Detection accuracy depends on lighting conditions and video quality.

## Contributions
Contributions are welcome! Please fork the repository and submit a pull request with your changes.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

---

MIT License

Copyright (c) 2025 Vijay Modi

## Acknowledgments
- [YOLOv8](https://github.com/ultralytics/ultralytics) for object detection.
- [SORT](https://github.com/abewley/sort) for real-time tracking.
- [OpenCV](https://opencv.org/) for video processing.

