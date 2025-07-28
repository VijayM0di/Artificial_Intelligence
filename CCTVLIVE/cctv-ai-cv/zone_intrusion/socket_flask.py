from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import cv2
import base64

app = Flask(__name__)
socketio = SocketIO(app)


# Function to send video frames and coordinates to the Frontend
def send_frame_data_to_frontend(frame, coordinates):
    # Convert the frame to base64 format
    _, buffer = cv2.imencode('.jpg', frame)
    frame_data = base64.b64encode(buffer).decode('utf-8')

    # Create a JSON object containing frame data and coordinates
    frame_packet = {
        'frame': frame_data,
        'coordinates': coordinates
    }

    emit('video_frame', frame_packet)


# Your OpenCV video streaming and violation detection logic here
def stream_video():
    cap = cv2.VideoCapture('your_video_source.mp4')
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Example coordinates for demonstration purposes
        coordinates = [
            {'x': 100, 'y': 200, 'width': 50, 'height': 50},
            {'x': 300, 'y': 150, 'width': 70, 'height': 70},
            # Add more object coordinates as needed
        ]

        send_frame_data_to_frontend(frame, coordinates)


@app.route('/')
def index():
    return render_template('index.html')


if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000)
