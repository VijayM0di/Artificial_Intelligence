import os
import shutil
import sys
import cv2
import requests
import json
from helpers.log_manager import error_logger, logger
from settings import CAMERA_RETRIEVAL_URL, CAMERA_RETRIEVAL_BY_IP_URL, access_token, RTSP_URL_PATTERN
from common import RequestMethod, API

frame_dir = 'frame_store'


if len(sys.argv) < 2:
    error_logger.error("Camera IP address not provided")
    sys.exit(-1)
camera_ip = sys.argv[1]

# camera_req = requests.post(CAMERA_RETRIEVAL_URL, json={"advancedSearch": {"fields": ["ipAddress"], "keyword": camera_ip}}, headers={'Authorization': f'Bearer {access_token}'})
#camera_req = API(RequestMethod.POST, CAMERA_RETRIEVAL_URL, json={"advancedSearch": {"fields": ["ipAddress"], "keyword": camera_ip}}, headers={'Authorization': f'Bearer {access_token}'})

camera_req = API(RequestMethod.GET, f"{CAMERA_RETRIEVAL_BY_IP_URL}/{camera_ip}", headers={'Authorization': f'Bearer {access_token}'})

logger.debug(f'Camera Res = {camera_req.status_code}')

if camera_req.status_code != 200:
    error_logger.error("There was some ERROR with the server!")
    sys.exit(-1)
camera_data = json.loads(camera_req.text)
if not camera_data:
    error_logger.error(f"Camera with IP {camera_ip} not found!!")
    sys.exit(-1)
    
logger.debug(f"Camera {camera_ip} Loaded!!")

RTSP_URL = RTSP_URL_PATTERN.format(username=camera_data['username'],
                                   password=camera_data['password'], 
                                   ipAddress=camera_ip, suffixRtspUrl=camera_data['urlSuffix'])


def get_frame(cam_ip, most_recent=True):
    cam_ip = cam_ip.replace('.', '-')
    files = os.listdir(frame_dir)
    frame_files = [file for file in files if file.startswith(f'frame-{cam_ip}')]
    if not frame_files:
        return None

    frame_numbers = [int(file.split('_')[1].split('.')[0]) for file in frame_files]
    if most_recent:
        frame_numbers.remove(max(frame_numbers))
        frame_numbers.remove(max(frame_numbers))
        latest_frame_number = max(frame_numbers)
    else:
        latest_frame_number = min(frame_numbers)
    latest_frame_filename = f'frame-{cam_ip}_{latest_frame_number}.jpg'
    return latest_frame_filename


def read_frames_and_put_in_queue():
    clear_old_frames()
    vs = cv2.VideoCapture(RTSP_URL)
    frame_count = 0

    # Loop until the end of the video stream
    while True:
        # Load the frame
        (frame_exists, frame) = vs.read()
        # Test if it has reached the end of the video
        if not frame_exists:
            vs = cv2.VideoCapture(RTSP_URL)
            continue
        else:
            frame_count += 1
            frame_filename = f'{frame_dir}/frame-{camera_ip.replace(".", "-")}_{frame_count}.jpg'
            cv2.imwrite(frame_filename, frame)
        if frame_count >= 150:
            frame_file = get_frame(most_recent=False, cam_ip=camera_ip)
            os.remove(f'{frame_dir}/{frame_file}')

    vs.release()


def clear_old_frames():
    shutil.rmtree(frame_dir)
    os.makedirs(frame_dir)


if __name__ == "__main__":
    read_frames_and_put_in_queue()
