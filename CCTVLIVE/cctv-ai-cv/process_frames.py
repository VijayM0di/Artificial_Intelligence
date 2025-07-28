import time
import cv2
import imutils
from read_from_rtsp import get_frame, frame_dir


def object_detection(camera_ip):
    frame_filename = get_frame(camera_ip)
    while frame_filename:
        frame = cv2.imread(f'{frame_dir}/{frame_filename}')
        if frame is None:
            frame_filename = get_frame(camera_ip)
            continue
        frame = imutils.resize(frame, width=int(720), height=405)
        cv2.imshow("Video", frame)
        time.sleep(0.3)
        frame_filename = get_frame(camera_ip)


if __name__ == "__main__":
    cam_ip = '192.168,141.228'
    object_detection(cam_ip)
