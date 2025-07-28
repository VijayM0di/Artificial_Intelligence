# Audio-Analysis-and-Recording-System
This project allows users to record audio, analyze its waveform and frequency spectrum, and visualize the results.

## Features
- Record audio using a microphone.
- Visualize the waveform of recorded or loaded audio files.
- Display the frequency spectrum of the audio signal.

## Requirements
The following libraries are required to run the project:
- `numpy`
- `matplotlib`
- `scipy`
- `pyaudio`

Install dependencies using:
```bash
pip install -r requirements.txt
```

## Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/Quantmbot-AI/Audio-Analysis.git
   cd Audio-Analysis
   ```

2. Record audio or analyze an existing `.wav` file.

### Recording Audio
To record audio, set the desired file path and recording duration, then run:
```python
file_path = r"output_audio.wav"
record_audio(file_path, record_seconds=60)
```

### Analyzing Audio
To analyze an existing `.wav` file:
```python
file_path = r"input_audio.wav"
analyze_audio(file_path)
```

3. Recorded audio will be saved to the specified file path, and plots for the waveform and spectrum will be displayed.

## Customization
- **Recording Duration:** Adjust the `record_seconds` parameter in `record_audio`.
- **Sampling Rate:** Modify the `sample_rate` parameter in `record_audio`.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

---

### Notes
- The system uses PyAudio for recording, which requires microphone access.
- Use `.wav` files for analysis to ensure compatibility.

