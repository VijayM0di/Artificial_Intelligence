import os
import time
import requests

from enum import Enum
from dotenv import load_dotenv
from configparser import ConfigParser
from helpers.log_manager import error_logger, logger

load_dotenv()

BACKEND_URL = os.environ.get('BACKEND_URL')
username = os.environ.get('BACKEND_USER')
password = os.environ.get('BACKEND_PASSWORD')
tenant = os.environ.get('BACKEND_TENANT')
LOGIN_URL = f'{BACKEND_URL}/api/tokens/encrypted-ai-services'

config_file_name = 'api_token.ini'
config = ConfigParser()
config.read(config_file_name)


def generate_token():
    payload = {'email': username, 'password': password}
    headers = {'tenant': tenant}
    response = requests.request("POST", LOGIN_URL,
                                headers=headers, json=payload)
    if response.status_code == 200:
        response = response.json()
        return response['token']
    else:
        return ''


class RequestMethod(Enum):
    GET = 'GET'
    POST = 'POST'


def get_header():
    access_token = str(config.get("INFO", "token"))
    return {'Authorization': f'Bearer {access_token}'}


def API(method: RequestMethod, url: str, headers=None, data=None, json=None, auth=None):
    if not isinstance(method, RequestMethod):
        raise Exception("Invalid API method name provided! should be GET or POST, use RequestMethod from Common file")

    headers = get_header() if headers else None
    retries = 3
    response = None
    for _ in range(retries):
        response = requests.request(method.value, url, headers=headers, data=data, json=json, auth=auth)
        if response.status_code == 401:
            new_token = generate_token()
            if not new_token and _ == retries - 1:
                retries -= 1
                error_logger.error(f'Error fetching new JWT token! Retry {_}')
            elif new_token:
                logger.debug("New JWT token generated!")
                config.set("INFO", "token", new_token)
                with open(config_file_name, 'w') as config_file:
                    config.write(config_file)
                headers = {'Authorization': f'Bearer {new_token}'}
            time.sleep(2)
            continue
        else:
            break
    return response


if __name__ == "__main__":

    token = str(config.get("INFO", "token"))
    if not token:
        token = generate_token()
        config.set("INFO", "token", token)
        with open(config_file_name, 'w') as config_file:
            config.write(config_file)

    res = API(RequestMethod.GET, f'{BACKEND_URL}/api/v1/common/list-deleted-camera-usecases',
              headers={'Authorization': f'Bearer {token}'})
    print('Res = ', res.json())

    # preset_payload = {"productId": "bed46883-97d6-4ff6-872a-df9c534d5c87",
    #                  "advancedFilter": {
    #                      "field": "IsDefault",
    #                      "operator": "eq",
    #                      "value": True
    #                  },
    #                  "pageNumber": 0,
    #                  "pageSize": 9999,
    #                  "orderBy": ["id"]
    #                  }

    # PRESET_RETRIEVAL_URL = f'{BACKEND_URL}/api/v1/camerapresets/search'
    # CAMERA_RETRIEVAL_URL = f'{BACKEND_URL}/api/v1/cameraproducts/search'
    # CAMERA_RETRIEVAL_BY_IP_URL = f'{BACKEND_URL}/api/v1/cameraproducts/by-camera-ip'
    # camera_ip = '192.168.141.2'
    # camera_req = API(RequestMethod.GET, CAMERA_RETRIEVAL_BY_IP_URL+'/{camera_ip}', headers={'Authorization': f'Bearer {token}'})
    # if camera_req.status_code != 200:
    #    print("There was some ERROR with the server!")
    # camera_data = json.loads(camera_req.text)

    # print('Res = ', camera_data)
