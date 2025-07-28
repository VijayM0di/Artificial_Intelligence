# Vijay-Vehicle-Speed-and-Multi-Zone-Detection-System
This project analyzes vehicle movement in a video, detects vehicles, tracks their speed, identifies entry zones, and flags violations such as re-entry into zones using YOLO and other computer vision techniques.

## Features

- Detects and tracks vehicles in a video.
- Calculates vehicle speed based on pixel movement and time.
- Flags vehicles exceeding the speed limit as violations.
- Identifies and monitors multi-zone entries.
- Alerts for re-entry violations into specific zones.
- Outputs an annotated video with bounding boxes, speed labels, zone annotations, and violation markers.

## Requirements

The following libraries are required to run the project:

- `numpy`
- `opencv-python`
- `imutils`
- `ultralytics`

Install dependencies using:

```bash
pip install -r requirements.txt
```

## Usage

### Vehicle Speed Detection

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

### Multi-Zone Detection and Violation Alerts

1. Set up the input video path in the script:

   ```python
   video_file_path = r"C:\path\to\your_video.mp4"
   ```

2. Configure the zones by defining a rectangle and deriving zones:

   ```python
   rectangle = [(90, 10), (650, 300)]
   zones = derive_zones(rectangle)
   ```

3. Run the script to process the video and monitor zones:

   ```bash
   python multi_zone_detection.py
   ```

4. View the live video feed with annotations or save the output for analysis.

## Customization

- **Source Video:** Change `SOURCE_VIDEO_PATH` or `video_file_path` in the scripts to your desired video file.
- **Model Configuration:** Modify `MODEL_NAME` to use a different YOLO model.
- **Speed Threshold:** Adjust the violation speed threshold in the vehicle speed detection script to change the criteria for flagging violations.
- **Polygon Zone:** Update the `SOURCE` variable or `rectangle` to define custom detection zones in the video.
- **Zone Colors:** Customize the colors for zone annotations (`COLOR_GREEN`, `COLOR_RED`, etc.) in the multi-zone detection script.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

### Notes

- Ensure the input video resolution and frame rate are suitable for accurate speed and zone analysis.
- Use a pre-trained YOLO model compatible with the `ultralytics` library.
- Test with various detection zones to optimize performance for your specific use case.

