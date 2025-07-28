# vijay-U-turn Detection
# U-turn Detection

## Overview

The U-turn Detection project uses YOLOv8 (You Only Look Once) and advanced perspective transformation techniques for real-time vehicle behavior monitoring. The system detects vehicles and tracks their movement to identify behaviors such as U-turns, reversing, and speed violations in video feeds.

## Features

- **Real-Time Detection**: Detects and tracks vehicles using the YOLOv8 model.
- **Behavior Analysis**: Identifies U-turns, reversing, and speed violations.
- **Perspective Transformation**: Converts detected vehicle coordinates to a top-down view for accurate analysis.
- **Customizable Parameters**: Configure detection thresholds and area boundaries.
- **Visual Overlays**: Displays bounding boxes, traces, and annotations directly on the video feed.

## Requirements

### Hardware

- A computer with a powerful GPU (recommended: NVIDIA RTX series).

### Software

- Python 3.8 or later
- Required Python libraries:
  - `numpy`
  - `opencv-python`
  - `opencv-python-headless`
  - `tqdm`
  - `supervision`
  - `ultralytics`
  - `collections`

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/Quantmbot-AI/vijay_U_turn.git
   cd vijay_U_turn
   ```

2. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Download the YOLOv8 model weights and place them in the project directory. You can download it from [Ultralytics](https://github.com/ultralytics/ultralytics).

4. Prepare your video file or ensure your webcam is accessible.

## File Structure

```
u-turn-detection/
|-- main.py                # Main program file
|-- requirements.txt       # Dependencies
|-- README.md              # Project description and instructions
```

## Usage

1. Modify the `SOURCE_VIDEO_PATH` in `main.py` to your video file path:

   ```python
   SOURCE_VIDEO_PATH = r"path_to_your_video.mp4"
   ```

2. Run the program:

   ```bash
   python main.py
   ```

3. Observe the video feed with overlays showing detected vehicles, IDs, behaviors, and speed violations. The output video will be saved as `vehicles-resulta.mp4`.

## Customization

- **Detection Area**: Update the `SOURCE` variable in `main.py` to define a custom area of interest:

  ```python
  SOURCE = np.array([
      [100, 200],
      [1800, 200],
      [1800, 950],
      [100, 950]
  ])
  ```

- **Confidence Threshold**: Adjust the confidence threshold for detections in `main.py`:

  ```python
  CONFIDENCE_THRESHOLD = 0.3
  ```

## Known Issues

- Performance may degrade on systems without a GPU.
- Accuracy may vary with poor lighting or low-quality video feeds.

## Contributions

Contributions are welcome! Please fork the repository and submit a pull request with your changes.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

---

MIT License

Copyright (c) 2025 Vijay Modi

## Acknowledgments

- [YOLOv8](https://github.com/ultralytics/ultralytics) for object detection.
- [Supervision](https://github.com/roboflow/supervision) for video frame processing.
- [OpenCV](https://opencv.org/) for video manipulation.

