import os
import sys
import ast
from dotenv import load_dotenv
from helpers.log_manager import error_logger, logger
from common import RequestMethod, API, generate_token, config_file_name, config

load_dotenv()

BACKEND_URL = os.environ.get('BACKEND_URL')
username = os.environ.get('BACKEND_USER')
password = os.environ.get('BACKEND_PASSWORD')
tenant = os.environ.get('BACKEND_TENANT')
sudo_password = os.environ.get('SUDO_PASSWORD')

project_directory = os.environ.get('PROJECT_DIRECTORY')
project_python_directory = os.environ.get('PROJECT_PYTHON_DIRECTORY')
environment_directory = os.environ.get('ENVIRONMENT_DIRECTORY')

service_command = f'{project_python_directory} {project_directory}'

LOGIN_URL = f'{BACKEND_URL}/api/tokens/encrypted-ai-service'
USECASE_RETRIEVAL_URL = f'{BACKEND_URL}/api/v1/camerausecases/search'
CAMERA_RETRIEVAL_URL = f'{BACKEND_URL}/api/v1/cameraproducts/search'
USE_CASES_RETRIEVAL_URL = f'{BACKEND_URL}/api/v1/usecases/search'
VIOLATION_FEED_CREATION_URL = f'{BACKEND_URL}/api/v1/violationfeeds'
ZONE_RETRIEVAL_URL = f'{BACKEND_URL}/api/v1/zones/search'
PRESET_RETRIEVAL_URL = f'{BACKEND_URL}/api/v1/camerapresets/search'
CAMERA_RETRIEVAL_BY_IP_URL = f'{BACKEND_URL}/api/v1/cameraproducts/by-camera-ip'
OVERSPEED_LIMIT_URL = f"{BACKEND_URL}/api/v1/settings/UseCase.Violation.OverspeedLimit"

try:
    access_token = str(config.get("INFO", "token"))
    if not access_token:
        raise Exception()
except:
    access_token = generate_token()
    config.set("INFO", "token", access_token)
    with open(config_file_name, 'w') as config_file:
        config.write(config_file)

MIN_VIDEO_THRESHOLD_URL = f'{BACKEND_URL}/api/v1/settings/UseCase.Violation.MinimumThreshold'
MIN_VIDEO_THRESHOLD_RESPONSE = API(RequestMethod.GET, MIN_VIDEO_THRESHOLD_URL,
                                   headers={'Authorization': f'Bearer {access_token}'})
# print(f'{MIN_VIDEO_THRESHOLD_RESPONSE=}')

if MIN_VIDEO_THRESHOLD_RESPONSE.status_code == 200:
    MIN_VIDEO_THRESHOLD = MIN_VIDEO_THRESHOLD_RESPONSE.json()
    MIN_VIDEO_THRESHOLD = int(MIN_VIDEO_THRESHOLD.get('value'))
else:
    error_logger.error('Minimum Violation Threshold not found in settings')
    sys.exit(-1)
    
MAX_VIDEO_THRESHOLD_URL = f'{BACKEND_URL}/api/v1/settings/UseCase.Violation.MaximumThreshold'
MAX_VIDEO_THRESHOLD_RESPONSE = API(RequestMethod.GET, MAX_VIDEO_THRESHOLD_URL,
                                   headers={'Authorization': f'Bearer {access_token}'})
# print(f'{MAX_VIDEO_THRESHOLD_RESPONSE=}')

if MAX_VIDEO_THRESHOLD_RESPONSE.status_code == 200:
    MAX_VIDEO_THRESHOLD = MAX_VIDEO_THRESHOLD_RESPONSE.json()
    MAX_VIDEO_THRESHOLD = int(MAX_VIDEO_THRESHOLD.get('value'))
else:
    error_logger.error('Maximum Violation Threshold not found in settings')
    sys.exit(-1)

FPS = int(os.environ.get('RTSP_STREAM_FPS', 25))
RTSP_URL_PATTERN = 'rtsp://{username}:{password}@{ipAddress}{suffixRtspUrl}'

RABBIT_MQ_HOST = os.environ.get('RABBIT_MQ_HOST')
RABBIT_MQ_USERNAME = os.environ.get('RABBIT_MQ_USERNAME')
RABBIT_MQ_PASSWORD = os.environ.get('RABBIT_MQ_PASSWORD')
RABBIT_MQ_VHOST = os.environ.get('RABBIT_MQ_VHOST')

RABBIT_MQ_EXCHANGE_TYPE = os.environ.get('RABBIT_MQ_EXCHANGE_TYPE')
RABBIT_MQ_EXCHANGE_NAME = os.environ.get('RABBIT_MQ_EXCHANGE_NAME')
RABBIT_MQ_TTL = int(os.environ.get('RABBIT_MQ_TTL'))

RABBIT_MQ_QUEUE_NAME = os.environ.get('RABBIT_MQ_QUEUE_NAME')
RABBIT_MQ_LIVE_QUEUE_NAME = os.environ.get('RABBIT_MQ_VIOLATION_QUEUE')

MINIO_HOST = os.environ.get('MINIO_HOST')
MINIO_ACCESS_KEY = os.environ.get('MINIO_ACCESS_KEY')
MINIO_SECRET_KEY = os.environ.get('MINIO_SECRET_KEY')
MINIO_BUCKET_NAME = os.environ.get('MINIO_BUCKET_NAME')
MINIO_PATH = os.environ.get('MINIO_PATH')
                   
CAMERA_URL_TO_PRESET = os.environ.get('CAMERA_URL_TO_PRESET')
CAMERA_URL_TO_PRESET = ast.literal_eval(CAMERA_URL_TO_PRESET)                       
               
USECASE_ABBR = os.environ.get('USECASE_ABBR')
USECASE_ABBR = ast.literal_eval(USECASE_ABBR)
                     
USECASE_PATH_NAME = os.environ.get('USECASE_PATH_NAME')
USECASE_PATH_NAME = ast.literal_eval(USECASE_PATH_NAME)

USECASE_PATH_ID = os.environ.get('USECASE_PATH_ID')
USECASE_PATH_ID = ast.literal_eval(USECASE_PATH_ID)

CALIBRATION_IMG_PATH = 'img/static_frame_from_video_{ip}.jpg'

HEIGHT_OG = int(os.environ.get('HEIGHT_OG'))
WIDTH_OG = int(os.environ.get('WIDTH_OG'))
SIZE_FRAME = int(os.environ.get('SIZE_FRAME'))
PUB_PORT = int(os.environ.get('PUB_PORT'))

UC_TRAFFIC_RULE_SPEED_LIMIT = os.environ.get('UC_TRAFFIC_RULE_SPEED_LIMIT')
UC_WRONG_TURN_MIN_RATIO = float(os.environ.get('UC_WRONG_TURN_MIN_RATIO'))

camera_req = API(RequestMethod.POST, CAMERA_RETRIEVAL_URL, json={}, headers={'Authorization': f'Bearer {access_token}'})
                           
if camera_req.status_code != 200:
    error_logger.error("There was some ERROR with the server!")
    sys.exit(-1)

cameras = camera_req.json()
camera_data = None
if cameras['data']:
    camera_data = cameras['data']

RTSP_TO_PORT = {}
for camera in camera_data:
    camera_ip = camera['ipAddress'].split(':')[0]
    RTSP_URL = RTSP_URL_PATTERN.format(username=camera['username'],
                                       password=camera['password'], 
                                       ipAddress=camera_ip, suffixRtspUrl=camera['urlSuffix'])
    RTSP_TO_PORT[RTSP_URL] = int('60' + camera_ip.split('.')[-1])

CAMERA_PORTS = list(RTSP_TO_PORT.values())
RTSP_URLS = list(RTSP_TO_PORT.keys())
logger.debug("CUDA_VISIBLE_DEVICES: " + os.getenv('CUDA_VISIBLE_DEVICES'))
