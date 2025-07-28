# Vijay-Voice-Navigation-Script

This Python script allows you to use voice commands to open specific URLs in your default web browser. It uses the `speech_recognition` library to recognize spoken commands and map them to corresponding URLs.

## Features
- Recognizes predefined voice commands.
- Opens URLs in the default web browser based on the command.
- Handles common errors during speech recognition.
- Includes an exit command for convenience.
- Works with any standard microphone and the Google Speech Recognition API.
- Easy to extend with new commands and URLs.

## Prerequisites
1. Python 3.6 or later.
2. A working microphone.
3. Internet access for the script to connect to the speech recognition service.

## Installation

### Step 1: Clone the repository or copy the script
Clone the repository or copy the script to your local machine.

### Step 2: Install dependencies
Use the following command to install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

1. Run the script:
   ```bash
   python Voice_navigation.py
   ```
2. Speak a command (e.g., "open Google", "open YouTube").
3. The script will open the corresponding URL in your default browser.
4. Say "exit" to stop the program.

## Customization
- To add new commands, edit the command-to-URL mapping in the script.

## License
This project is licensed under the MIT License. See the LICENSE file for details.
