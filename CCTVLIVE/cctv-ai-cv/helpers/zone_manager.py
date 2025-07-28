import json

import requests

from helpers.log_manager import error_logger, logger
from settings import ZONE_RETRIEVAL_URL, access_token
from common import RequestMethod, API


def fetch_zone_camera_use_case(camera_id, use_case_id):
    zone_payload = {"productId": camera_id,
                    "advancedFilter": {
                        "field": "UseCaseId",
                        "operator": "eq",
                        "value": use_case_id
                    },
                    "pageNumber": 0,
                    "pageSize": 9999,
                    "orderBy": ["id"]
                    }
                    
    # zone_req = requests.post(ZONE_RETRIEVAL_URL, json=zone_payload, headers={'Authorization': f'Bearer {access_token}'})
    zone_req = API(RequestMethod.POST, ZONE_RETRIEVAL_URL, json=zone_payload, headers={'Authorization': f'Bearer {access_token}'})
    
    
    if zone_req.status_code != 200:
        error_logger.error(f"Zone retrieval from DB failed!!")
    else:
        logger.debug(f"Zone retrieval from DB successful!!")
    zone_data = zone_req.json()
    zones = zone_data['data']
    if zones:
        zone = zones[0]['metaData']
        zone = json.loads(zone)
        zone = zone['zone']
        zone_id = zones[0]['id']
    else:
        zone = None
        zone_id = None
    return zone, zone_id
