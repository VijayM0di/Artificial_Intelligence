# Vijay-Safety-Kit-Detection-System
This project uses deep learning to detect safety equipment in video feeds, identifying whether individuals are wearing proper personal protective equipment (PPE) such as hardhats, masks, and safety vests.

## Features
- Real-time detection of safety kits and PPE compliance.
- Processes video input and highlights detected items with bounding boxes.
- Generates output video with detection information overlayed.

## Requirements
The following libraries are required to run the project:
- `ultralytics`
- `opencv-python`
- `cvzone`

Install dependencies using:
```bash
pip install -r requirements.txt
```

## Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/Quantmbot-AI/Safety-Kit-Detection.git
   cd Safety-Kit-Detection
   ```

2. Place your video file in the project directory.

3. Run the script:
   ```bash
   python safety_kit_detection.py
   ```

4. Processed video output will be saved as `test_case-1.mp4`.

## Customization
- **Input Video:** Change the `cap` variable to point to your input video.
- **Output Path:** Modify the `output_path` variable to set a custom output file name.

## Example
```python
cap = cv2.VideoCapture("input_video.avi")
output_path = "output_video.mp4"
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
```

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

---

### Notes
- The system classifies objects into categories such as Hardhat, Mask, Safety Vest, and identifies unsafe conditions.
- Press `Q` to stop the video playback during processing.

