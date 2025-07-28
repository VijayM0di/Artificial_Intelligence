from datetime import timedelta

from minio import Minio

import glob
import os


def upload_local_directory_to_minio(local_path, bucket_name, minio_path):
    assert os.path.isdir(local_path)

    for local_file in glob.glob(local_path + '/**'):
        local_file = local_file.replace(os.sep, "/")  # Replace \ with / on Windows
        if not os.path.isfile(local_file):
            upload_local_directory_to_minio(
                local_file, bucket_name, minio_path + "/" + os.path.basename(local_file))
        else:
            remote_path = os.path.join(
                minio_path, local_file[1 + len(local_path):])
            remote_path = remote_path.replace(
                os.sep, "/")  # Replace \ with / on Windows
            client.fput_object(bucket_name, remote_path, local_file)


client = Minio('192.168.56.41:35418', access_key='cbcctvaiapp',
               secret_key='CBCCTVAiApp$#@!$@nakieteyq983@$!#@872361#@!$jqsftyhkak@18273', secure=False)
# client.make_bucket('pythonbucket', location='us-west-1')

buckets = client.list_buckets()
for bucket in buckets:
    print('bucket: ', bucket)

data_sample = open('file.txt', 'w')
data_sample.write('This is some text')
data_sample.close()

# client.fput_object('cb-cctvai', 'bucket/contents/file.txt','file.txt')
# upload_local_directory_to_minio('videos', 'cb-cctvai', 'bucket/videos')


objects = client.list_objects('cb-cctvai', prefix='bucket/videos/', recursive=True)
presigned_urls = []

for obj in objects:
    # print('objects: ', obj.object_name, obj.size)
    object_name = obj.object_name
    if not object_name.endswith('mp4'):
        continue
    expiration_time = timedelta(seconds=604800)
    presigned_url = client.presigned_get_object('cb-cctvai', object_name, expiration_time)
    if presigned_url:
        presigned_urls.append(presigned_url)
        print(presigned_url)
