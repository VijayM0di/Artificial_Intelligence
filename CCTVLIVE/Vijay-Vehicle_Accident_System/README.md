# Vijay-Vehicle_Accident_System
This project processes video frames to detect vehicle accidents using a pre-trained model. It highlights detected accidents by drawing bounding boxes around them in the video and saves the processed video with these highlights.

## Features
- Real-time traffic accident detection.
- Processes video input and highlights accident-prone areas.
- Generates output video with bounding boxes drawn around detected accidents.

## Requirements
The following libraries are required to run the project:
- `torch`
- `torchvision`
- `transformers`
- `numpy`
- `Pillow`
- `opencv-python`

Install dependencies using:
```bash
pip install -r requirements.txt
```

## Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/Quantmbot-AI/Vehicle-Accident-Detection.git
   cd Vehicle-Accident-Detection
   ```

2. Place your video file in the project directory.

3. Run the script:
   ```bash
   python Accident.py
   ```

4. Processed video output will be saved as `output_video5.mp4`.

## Customization
- **Input Video:** Change the `video_path` variable to point to your input video.
- **Output Path:** Modify `output_path` to set a custom output file name.

## Example
```python
video_path = "cr.mp4"
output_path = "output_video5.mp4"
detect_and_save_accidents(video_path, output_path)
```

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

---

### Notes
- Lowered detection threshold in the script ensures higher sensitivity for identifying accidents.
- You can press `Q` to stop the video playback during processing.

