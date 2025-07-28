# cctv-ai-cv

CCTV-AI-CV is a computer vision project for real-time video analytics using deep learning. It is designed for applications such as surveillance, safety monitoring, and automated event detection in CCTV camera feeds.

## Features
- Real-time video stream processing
- Object detection and tracking
- Zone intrusion and safety monitoring
- Modular design for easy extension (add new models or analytics)
- Docker support for deployment

## Project Structure
- `models/`, `ppt_models/`, etc.: Pretrained models and analytics modules
- `stream_fetcher.py`, `stream_distributor.py`: Video stream handling
- `zone_intrusion/`, `waiting_area/`, etc.: Specialized analytics modules
- `requirements.txt`, `uat_requirements.txt`, `prod_requirements.txt`: Dependency files for different environments
- `Dockerfile`: For containerized deployment

## Requirements
- Python 3.8+
- See `requirements.txt` for dependencies
- (Optional) Docker for containerized deployment

Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/cctv-ai-cv.git
   cd cctv-ai-cv
   ```
2. Configure your video sources and analytics modules in `settings.py` or relevant config files.
3. Run the main service (example):
   ```bash
   python camera_starter_service.py
   ```
4. For Docker deployment:
   ```bash
   docker build -t cctv-ai-cv .
   docker run -it --rm cctv-ai-cv
   ```

## Customization
- Add new analytics modules in the respective folders
- Update configuration in `settings.py`
- Use different requirements files for UAT/production as needed

## License
This project is licensed under the MIT License. See the LICENSE file for details.

---

### Notes
- For detailed documentation, see `cctv-ai-cv-docs.docx` in the project root.
- Example payloads and configs are provided in the `payload.json` and `camera-preset.json` files.
