#!/usr/bin/env python
import base64
import json
import os

import pika

creds = pika.PlainCredentials('cctvai_user', 'XTbcT9QbRD3yB27')
connection = pika.BlockingConnection(
    pika.ConnectionParameters(host='192.168.53.111', virtual_host='cctvai', credentials=creds))
channel = connection.channel()

channel.queue_declare(queue='testQueue', durable=True)
# channel.queue_declare(queue='cctvaiQueue', durable=True)

from pyrabbit.api import Client
# cl = Client(host='192.168.53.111:15672', user='cctvai_user', passwd='XTbcT9QbRD3yB27')
# q_names = [q['name'] for q in cl.get_queues('cctvai')]
# print(q_names)
count = 0
dir_path = '/media/AISTORE/IQRAFOLDER/Code/outputs_AI_CCTV/output_frames_intrusion2'
for item in sorted(os.listdir(dir_path)):
    if item.endswith('.json'):
        json_path = os.path.join(dir_path, item)
        img_filename = item.replace('.json', '.jpg')
        img_path = os.path.join(dir_path, img_filename)
        if os.path.exists(img_path):
            with open(img_path, 'rb') as img_file:
                img_data = img_file.read()
                img_base64 = base64.b64encode(img_data).decode('utf-8')

            with open(json_path, 'r') as json_file:
                json_data = json.load(json_file)

            combined_data = {
                'image': img_base64,
                'metadata': json_data
            }

            combined_data_bytes = json.dumps(combined_data).encode('utf-8')

            channel.basic_publish(exchange='', routing_key='testQueue', body=combined_data_bytes)
            print(count)
            count += 1

connection.close()
