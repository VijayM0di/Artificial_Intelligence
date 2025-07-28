from minio import Minio
from settings import MINIO_SECRET_KEY, MINIO_HOST, MINIO_ACCESS_KEY

client = Minio(MINIO_HOST, access_key=MINIO_ACCESS_KEY,
               secret_key=MINIO_SECRET_KEY, secure=False)
