# Vehicle Detection System

This project uses deep learning to detect vehicles in video feeds. It processes video frames to identify and highlight vehicles, providing a count of detected vehicles.

## Features
- Real-time vehicle detection.
- Processes video input and highlights detected vehicles with bounding boxes.
- Generates output video with vehicle information overlayed.

## Requirements
The following libraries are required to run the project:
- `ultralytics`
- `opencv-python`

Install dependencies using:
```bash
pip install -r requirements.txt
```

## Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/Quantmbot-AI/Vehicle-Detection.git
   cd Vehicle-Detection
   ```

2. Place your video file in the project directory.

3. Run the script:
   ```bash
   python vehicle_detection.py
   ```

4. Processed video output will be saved as `vehicle_output.avi`.

## Customization
- **Input Video:** Change the `cap` variable to point to your input video.
- **Output Path:** Modify the `out` variable to set a custom output file name.

## Example
```python
cap = cv2.VideoCapture("cars.mp4")
out = cv2.VideoWriter('vehicle_output.avi', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))
```

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

---

### Notes
- You can press `Q` to stop the video playback during processing.

