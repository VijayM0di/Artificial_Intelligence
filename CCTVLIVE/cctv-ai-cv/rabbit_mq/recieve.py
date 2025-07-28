#!/usr/bin/env python
import json

import pika, sys, os
import requests
from dotenv import load_dotenv


def main():
    creds = pika.PlainCredentials('cctvai_user', 'XTbcT9QbRD3yB27')
    connection = pika.BlockingConnection(
        pika.ConnectionParameters(host='192.168.20.230', virtual_host='cctvai', credentials=creds))
    channel = connection.channel()
    queue = 'cctvaiLiveQueue'
    # queue = 'testQueue'
    channel.queue_declare(queue=queue, durable=True
                          , arguments={'x-message-ttl': 90000}
                          )
    test = json.dumps({'body': 'Hello there!'}).encode('utf-8')
    
    
    #USECASE_PATH_ID = {
    #    '19fcf595-91e1-4503-b7d2-695322e56d86': 'four_vehicles_per_crane/src/vehicles_in_close_proximity.py',
    #    '39796858-3a2e-4f58-a8a6-5e5d08211b88': 'people_under_suspended_load/src/people_under_suspended_load.py',
    #    '841975bc-3271-4698-aa82-6bc5c029424a': 'PPE_Kit/src/ppe_kit_detection.py',
    #    'b1ac6b8c-e239-4e98-884d-fc84a850fb56': 'zone_intrusion/src/zone_intrusion_detection.py',
    #    'cd088b7d-5646-4a2c-a8dc-c9d01b6b02eb': 'water_edge/src/water_edge_detection.py',
    #    'd9960718-dfa3-45e3-8b3c-bf9c4f2c0d52': 'people_close_to_moving_objects/src/people_close_to_moving_objects.py',
    #    'a3ee5083-1ceb-7f31-6269-c36917df8dc3': 'traffic_rules/src/no_entry_detection.py',
    #    'afc4bf59-a4c8-505d-6fd4-3d2cbba69fa1': 'traffic_rules/src/illegal_parking_violation_detection.py',
    #    '3a7153f9-f9d1-c86c-3db8-786bcd6abb7b': 'traffic_rules/src/over_speeding_detection.py',
    #    '5b9795ed-e39b-f317-4c8a-17c324c1603e': 'traffic_rules/src/wrong_u_turn_detection.py',
    #    'f1cb77d8-7eef-4216-b995-8bcefc967861': 'waiting_area/src/waiting_area_detection.py'
    #    }
        
    USECASE_PATH_ID = os.environ.get('USECASE_PATH_ID')
    USECASE_PATH_ID = ast.literal_eval(USECASE_PATH_ID)
        
    for usecase in USECASE_PATH_ID:
        test = json.dumps(json.dumps({'CameraId': '2b8849c3-54aa-4215-8a0c-6e187fb6bba2', 'UseCaseId':
            usecase, 'DurationInSeconds': 20})).encode('utf-8')
        # channel.basic_publish(exchange='', routing_key=queue, body=test)
    load_dotenv()

    BACKEND_URL = os.environ.get('BACKEND_URL')
    REQUEST_LIVE_FEED_URL = f'{BACKEND_URL}/api/v1/common/request-live-feed/2b8849c3-54aa-4215-8a0c-6e187fb6bba2/b1ac6b8c-e239-4e98-884d-fc84a850fb56'
    LOGIN_URL = f'{BACKEND_URL}/api/tokens'
    username = os.environ.get('BACKEND_USER')
    password = os.environ.get('BACKEND_PASSWORD')
    tenant = os.environ.get('BACKEND_TENANT')
    
    
    access_token_req = requests.post(LOGIN_URL, headers={'tenant': tenant}, json={'email': username, 'password': password})
    access_token = access_token_req.json()
    access_token = access_token['token']
    
    
    live_feed_req = requests.get(REQUEST_LIVE_FEED_URL, headers={'Authorization': f'Bearer {access_token}'})
    
    
    print(live_feed_req.json())
    all_messages_checked = False
    match_found = False
    while not all_messages_checked:
        method_frame, header_frame, body = channel.basic_get(queue=queue, auto_ack=False)
        if method_frame:
            message_body = body.decode()
            print(f"Received: {message_body}")
            message_body = json.loads(message_body)
            message_body = json.loads(message_body)
            camera_from_mq = message_body['CameraId']
            usecase_from_mq = message_body['UseCaseId']
    def callback(ch, method, properties, body):
        print(f" [x] Received {body}")
        body = body.decode()
        # body = json.loads(body)
        body = json.loads(body)
        print(body)

    channel.basic_consume(queue=queue, on_message_callback=callback, auto_ack=True)

    print(' [*] Waiting for messages. To exit press CTRL+C')
    channel.start_consuming()

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Interrupted')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)