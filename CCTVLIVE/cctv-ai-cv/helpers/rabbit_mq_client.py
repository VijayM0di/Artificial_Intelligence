import pika
from settings import RABBIT_MQ_USERNAME, RABBIT_MQ_PASSWORD, RABBIT_MQ_HOST, RABBIT_MQ_VHOST

creds = pika.PlainCredentials(RABBIT_MQ_USERNAME, RABBIT_MQ_PASSWORD)
connection = pika.BlockingConnection(
    pika.ConnectionParameters(host=RABBIT_MQ_HOST, virtual_host=RABBIT_MQ_VHOST, credentials=creds))
