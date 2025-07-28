import sys
import cv2
import zmq
import pickle
# from helpers.log_manager import error_logger
from settings import CAMERA_PORTS, RTSP_URLS


def fetch_and_push(rtsp_url, push_port):
    context = zmq.Context()
    socket = context.socket(zmq.PUSH)
    socket.connect(f"tcp://localhost:{push_port}")

    cap = cv2.VideoCapture(rtsp_url)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_bytes = pickle.dumps(frame)
        socket.send(frame_bytes)

    cap.release()
    socket.close()


if __name__ == '__main__':
    if len(sys.argv) < 2:
        error_logger.error("Camera IP address not provided")
        sys.exit(-1)
    camera_ip = sys.argv[1]
    for i, item in enumerate(RTSP_URLS):
        if camera_ip in item:
            fetch_and_push(rtsp_url=RTSP_URLS[i], push_port=CAMERA_PORTS[i])
