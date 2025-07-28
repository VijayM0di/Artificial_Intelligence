import json
import requests
from requests.auth import HTTPDigestAuth
from common import RequestMethod, API
from helpers.log_manager import error_logger, logger
from settings import PRESET_RETRIEVAL_URL, CAMERA_RETRIEVAL_BY_IP_URL, access_token, BACKEND_URL

def get_preset_url(camera_ip):
    
    req = API(RequestMethod.GET, f"{CAMERA_RETRIEVAL_BY_IP_URL}/{camera_ip}", headers={'Authorization': f'Bearer {access_token}'})
    
    if req.status_code != 200:
        error_logger.error("Camera product by camera IP request failed!")
        
    data = json.loads(req.text)
    if not data:
      error_logger.error(f"Camera with IP {camera_ip} not found!!")
      sys.exit(-1)
    
    presets = data["cameraPresets"]
    
    url = [i for i in presets if i['isDefault'] == True][0]['httpUrl']
    
    return url


def set_preset(camera_ip, camera_data):
    username = camera_data['username']
    password = camera_data['password']
    camera_id = camera_data['id']
    auth = HTTPDigestAuth(username, password)
    preset_id = get_default_preset_id_camera(camera_id)
    if preset_id:
    
        url = get_preset_url(camera_ip)
    
        url = url.replace(f'{username}:{password}@{camera_ip}', '{camera_ip}')
        
        url = url.format(camera_ip=camera_ip, preset_id=preset_id)
        logger.debug(f'Camera Preset Id Url: {url}')
    
        # preset = requests.get(f'http://{camera_ip}/stw-cgi/ptzcontrol.cgi?msubmenu=preset&action=control&Preset={preset_id}', auth=auth)
        #preset = API(RequestMethod.GET, url, headers={'Authorization': f'Bearer {access_token}'}, auth=auth)


def get_camera_calibration(camera_id):
    preset_payload = {"productId": camera_id,
                      "advancedFilter": {
                          "field": "IsDefault",
                          "operator": "eq",
                          "value": True
                      },
                      "pageNumber": 0,
                      "pageSize": 9999,
                      "orderBy": ["id"]
                      }
    # preset_req = requests.post(PRESET_RETRIEVAL_URL, json=preset_payload, headers={'Authorization': f'Bearer {access_token}'})
    preset_req = API(RequestMethod.POST, PRESET_RETRIEVAL_URL, json=preset_payload, headers={'Authorization': f'Bearer {access_token}'})    
    
    if preset_req.status_code != 200:
        error_logger.error("Preset calibration retrieval from DB failed!!")
    else:
        logger.debug("Preset calibration retrieval from DB successful!!")
    preset_data = preset_req.json()
    presets = preset_data['data']
    if presets:
        preset = presets[0]['metaData']
        preset = json.loads(preset)
        final_preset = [preset[2], preset[3], preset[1], preset[0]]
    else:
        final_preset = None
    return final_preset


def get_default_preset_id_camera(camera_id):
    preset_payload = {"productId": camera_id,
                      "advancedFilter": {
                          "field": "IsDefault",
                          "operator": "eq",
                          "value": True
                      },
                      "pageNumber": 0,
                      "pageSize": 9999,
                      "orderBy": ["id"]
                      }
    # preset_req = requests.post(PRESET_RETRIEVAL_URL, json=preset_payload, headers={'Authorization': f'Bearer {access_token}'})
    preset_req = API(RequestMethod.POST, PRESET_RETRIEVAL_URL, json=preset_payload, headers={'Authorization': f'Bearer {access_token}'})
    
    
    if preset_req.status_code != 200:
        error_logger.error("Preset data retrieval from DB failed!!")
    else:
        logger.debug("Preset data retrieval from DB successful!!")
    preset_data = preset_req.json()
    presets = preset_data['data']
    if presets:
        preset = presets[0]['presetId']
    else:
        preset = None
    return preset
