# Vehicle Speed Detection System

This project analyzes vehicle movement in a video, detects vehicles, tracks their speed, and flags speed violations using YOLO and other computer vision techniques.

## Features
- Detects and tracks vehicles in a video.
- Calculates vehicle speed based on pixel movement and time.
- Flags vehicles exceeding the speed limit as violations.
- Outputs an annotated video with bounding boxes, speed labels, and violation markers.

## Requirements
The following libraries are required to run the project:
- `numpy`
- `opencv-python`
- `supervision`
- `ultralytics`
- `tqdm`

Install dependencies using:
```bash
pip install -r requirements.txt
```

## Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/Quantmbot-AI/Vehicle-Speed-Detection.git
   cd Vehicle-Speed-Detection
   ```

2. Prepare your input video file and ensure the path is correctly set in the script:
   ```python
   SOURCE_VIDEO_PATH = "vehicles.mp4"
   ```

3. Run the script to process the video and detect speed violations:
   ```bash
   python vehicle_speed_detection.py
   ```

4. The output video with annotations will be saved as:
   ```
   vehicles-result.mp4
   ```

## Customization
- **Source Video:** Change `SOURCE_VIDEO_PATH` in the script to your desired video file.
- **Model Configuration:** Modify `MODEL_NAME` to use a different YOLO model.
- **Speed Threshold:** Adjust the violation speed threshold in the script to change the criteria for flagging violations.
- **Polygon Zone:** Update the `SOURCE` variable to define a custom detection zone in the video.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

---

### Notes
- Ensure the input video resolution and frame rate are suitable for accurate speed calculation.
- Use a pre-trained YOLO model compatible with the `ultralytics` library.

