# import websocket
import socketio
import threading
import time

sio = socketio.Client()

@sio.on("video_frame")
def on_message(message):
    print(f"Received message: {message}")

@sio.event
def connect():
    print(f"Connection established")


sio.connect('http://localhost:5000')
sio.wait()
