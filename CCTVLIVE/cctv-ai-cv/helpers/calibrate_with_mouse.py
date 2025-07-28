import sys
import cv2
import yaml
import imutils
import requests
import json
from requests.auth import HTTPDigestAuth
from helpers.log_manager import logger, error_logger
from settings import RTSP_URL_PATTERN, CAMERA_RETRIEVAL_BY_IP_URL, access_token, CAMERA_RETRIEVAL_URL, CAMERA_URL_TO_PRESET, SIZE_FRAME
from common import RequestMethod, API


with open('camera-preset.json', 'r') as json_file:
    cam_preset_data = json.load(json_file)

# Define the callback function that we are going to use to get our coordinates
def CallBackFunc(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        logger.debug("Left button of the mouse is clicked - position (", x, ", ",y, ")")
        list_points.append([x,y])
    elif event == cv2.EVENT_RBUTTONDOWN:
        logger.debug("Right button of the mouse is clicked - position (", x, ", ", y, ")")
        list_points.append([x,y])


if len(sys.argv) < 2:
    error_logger.error("Camera IP address not provided")
    sys.exit(-1)
camera_ip = sys.argv[1]
size_frame = SIZE_FRAME

auth = HTTPDigestAuth('admin', 'DPW.AI.cctv')

preset = None
preset_id = '1'

if camera_ip in CAMERA_URL_TO_PRESET:
    preset_id = CAMERA_URL_TO_PRESET[camera_ip]
    # preset = requests.get(f'http://{camera_ip}/stw-cgi/ptzcontrol.cgi?msubmenu=preset&action=control&Preset={preset_id}', auth=auth)
    preset = API(RequestMethod.GET, f'http://{camera_ip}/stw-cgi/ptzcontrol.cgi?msubmenu=preset&action=control&Preset={preset_id}', headers={'Authorization': f'Bearer {access_token}'}, auth=auth)
    
    
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
    
#cameras = camera_req.json()
#camera_data = None
#if cameras['data']:
#    camera_data = cameras['data'][0]
#else:
#    error_logger.error(f"Camera with IP {camera_ip} not found!!")
#    sys.exit(-1)

RTSP_URL = RTSP_URL_PATTERN.format(username=camera_data['username'],
                                   password=camera_data['password'], 
                                   ipAddress=camera_ip, suffixRtspUrl=camera_data['urlSuffix'])
                                   
vs = cv2.VideoCapture(RTSP_URL)
# vs = cv2.VideoCapture("video/people-under-suspended-load-demo.mp4")
# Loop until the end of the video stream
while True:
    # Load the frame and test if it has reache the end of the video
    (frame_exists, frame) = vs.read()
    if not frame_exists:
        continue
    frame = imutils.resize(frame, width=int(size_frame))
    img_path_1 = "img/static_frame_from_video_" + camera_ip.replace('.', '_') + ".jpg"
    cv2.imwrite(img_path_1,frame)
    break

# Create a black image and a window
windowName = 'MouseCallback'
cv2.namedWindow(windowName)


# Load the image 
img_path = "img/static_frame_from_video_"+camera_ip.replace('.', '_')+".jpg"
img = cv2.imread(img_path)

img_path = "" + img_path

# Get the size of the image for the calibration
width,height,_ = img.shape

# Create an empty list of points for the coordinates
list_points = list()

# bind the callback function to window
cv2.setMouseCallback(windowName, CallBackFunc)


if __name__ == "__main__":
    # Check if the 4 points have been saved
    while (True):
        cv2.imshow(windowName, img)
        if len(list_points) == 4:
            # Return a dict to the YAML file
            config_data = dict(
                image_parameters = dict(
                    p1=list_points[2],
                    p2 = list_points[3],
                    p3=list_points[1],
                    p4 = list_points[0],
                    width_og = width,
                    height_og = height,
                    img_path = img_path,
                    size_frame = size_frame,
                    ))
            # Write the result to the config file
            with open(f'conf/config_birdview_{camera_ip.replace(".", "_")}.yml', 'w') as outfile:
                yaml.dump(config_data, outfile, default_flow_style=False)
            break

        if cv2.waitKey(20) == 27:
            break
    cv2.destroyAllWindows()
