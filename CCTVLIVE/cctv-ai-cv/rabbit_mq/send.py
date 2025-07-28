#!/usr/bin/env python
import pika

creds = pika.PlainCredentials('cctvai_user', 'XTbcT9QbRD3yB27')
connection = pika.BlockingConnection(
    pika.ConnectionParameters(host='192.168.53.111', virtual_host='cctvai', credentials=creds))
channel = connection.channel()

channel.queue_declare(queue='cctvaiQueue')

channel.basic_publish(exchange='', routing_key='hello', body='Hello World!')
print(" [x] Sent 'Hello World!'")
connection.close()