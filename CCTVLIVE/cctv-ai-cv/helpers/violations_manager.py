import copy
import io
import shutil
import time
import uuid

import imageio
import numpy as np

from helpers.log_manager import logger, error_logger
from datetime import datetime, timedelta
import datetime as dt
import cv2
import ffmpeg
import requests
from helpers.json_encoder import CustomJSONEncoder
from helpers.minio_client import client
import base64
import json
import os
import pika
from concurrent.futures import ThreadPoolExecutor
from common import RequestMethod, API
from settings import (
    RABBIT_MQ_USERNAME,
    RABBIT_MQ_PASSWORD,
    RABBIT_MQ_HOST,
    RABBIT_MQ_VHOST,
    RABBIT_MQ_QUEUE_NAME,
    MINIO_BUCKET_NAME,
    MINIO_PATH,
    VIOLATION_FEED_CREATION_URL,
    access_token,
    FPS,
    MIN_VIDEO_THRESHOLD,
    MAX_VIDEO_THRESHOLD,
    RABBIT_MQ_LIVE_QUEUE_NAME,
    RABBIT_MQ_TTL,
    RABBIT_MQ_EXCHANGE_NAME,
    RABBIT_MQ_EXCHANGE_TYPE,
)

violationReferenceId = None
frame_sending_start_time = None
frame_override_time_interval = None
unq_req_id_mq = None


def clear_existing_frames(type_of_violation):
    if "traffic_rules" in type_of_violation:
        split_violation = type_of_violation.split("_")
        type_of_violation = split_violation[0] + "_" + split_violation[1]

    output_path_root = f"{type_of_violation}/output"
    output_path = f"{output_path_root}/frames"
    shutil.rmtree(output_path)
    shutil.rmtree(output_path_root)
    os.makedirs(output_path_root)
    os.makedirs(output_path)


def check_and_process_all_messages(channel, usecase_id, camera_id):
    global unq_req_id_mq
    all_messages_checked = False
    time_interval = None
    match_found = False
    # channel.basic_qos(prefetch_count=1, prefetch_size=0)

    while not all_messages_checked and not match_found:

        new_subscribe_queue = f"live.{camera_id}_{usecase_id}"
        method_frame, header_frame, body = channel.basic_get(
            queue="cctvai.liverequest", auto_ack=False
        )
        if method_frame:
            message_body = body.decode()
            print(f"Received: {message_body}")
            message_body = json.loads(message_body)
            message_body = json.loads(message_body)
            camera_from_mq = message_body["CameraId"]
            usecase_from_mq = message_body["UseCaseId"]
            unique_req_id_from_mq = message_body["unqReqId"]
            if camera_id == camera_from_mq and usecase_id == usecase_from_mq:
                time_interval = message_body["DurationInSeconds"]
                match_found = True
                unq_req_id_mq = unique_req_id_from_mq
                channel.basic_ack(delivery_tag=method_frame.delivery_tag)
            else:
                print("Message does not match")
        else:
            print("No message left in queue")
            all_messages_checked = True
    return match_found, time_interval


def generate_video_and_save(video_saving_queue, stop_event):

    logger.debug(f"Video Saving Queue = {video_saving_queue.qsize()} ")
    while not stop_event.is_set():

        task = video_saving_queue.get()  # Blocks until a task is available
        if task is None:  # None is the signal to stop the thread
            logger.debug("Waiting for the video saving queue thread.....")
            time.sleep(0.1)

        try:
            logger.debug("Starting video generation...")

            # Unpack task data
            frame = task.get("frame", [])
            output_path = task.get("output_path")
            output_path_mp4 = task.get("output_path_mp4")
            output_filename = task.get("output_filename")
            output_video_1_list = task.get("output_video_1_list", [])
            data_json = task.get("data_json")
            usecase_id = task.get("usecase_id")
            all_data = task.get("all_data")
            zone_id = task.get("zone_id")

            current_time = datetime.now()
            current_time = current_time.replace(tzinfo=dt.timezone.utc)
            current_time = current_time.isoformat()
            # writer = imageio.get_writer(output_path, fps=FPS)

            fourcc1 = cv2.VideoWriter_fourcc(*"MJPG")
            output_video_1 = cv2.VideoWriter(
                output_path,
                fourcc1,
                FPS,
                (int(frame.shape[1]), int(frame.shape[0])),
                True,
            )

            for item in output_video_1_list:
                output_video_1.write(item)
            output_video_1.release()

            if os.name == "posix":
                os.system("sync")
            input_stream = ffmpeg.input(output_path)
            output_stream = ffmpeg.output(
                input_stream, output_path_mp4, vcodec="libx264"
            )
            ffmpeg.run(output_stream)
            if os.path.exists(output_path_mp4):
                os.remove(output_path)
            remote_path = os.path.join(
                MINIO_PATH, output_filename.replace(".avi", ".mp4")
            )
            remote_path = remote_path.replace(
                os.sep, "/"
            )  # Replace \ with / on Windows
            try:
                client.fput_object(MINIO_BUCKET_NAME, remote_path, output_path_mp4)
                expiration_time = timedelta(seconds=604800)
                presigned_url = client.presigned_get_object(
                    MINIO_BUCKET_NAME, remote_path, expiration_time
                )
                if presigned_url:
                    all_data_json = json.dumps(all_data, cls=CustomJSONEncoder)
                    payload = {
                        "productId": data_json["camera_id"],
                        "usecaseId": usecase_id,
                        "zoneId": zone_id,
                        "violationType": "Violation",
                        "cameraUrl": data_json["camera_ip"],
                        "videoUrl": remote_path,
                        "metaViolation": all_data_json,
                        "metaCamera": "{}",
                        "metaZones": "{}",
                        "recordedOn": current_time,
                        "violationReferenceId": data_json["violationReferenceId"],
                    }

                    video_req = API(
                        RequestMethod.POST,
                        VIOLATION_FEED_CREATION_URL,
                        json=payload,
                        headers={"Authorization": f"Bearer {access_token}"},
                    )

                    if video_req.status_code != 200:
                        error_logger.error("There was some ERROR with the DB server!")
                    else:
                        logger.debug(f"VideoURL updated to DB: {presigned_url}")
            except Exception as e:
                error_logger.error(f"Upload Error: {output_path_mp4}")
            if os.path.exists(output_path_mp4):
                os.remove(output_path_mp4)

        except Exception as e:
            logger.debug(f"Error in video generation or API sending: {e}")
        finally:
            logger.debug("Video saving task completed....")


def write_video_upload_minio(
    frame,
    data_json,
    output_video_1_list,
    violation,
    type_of_violation,
    start_time,
    usecase_id,
    all_data,
    zone_id=None,
    output_video_frame_list=[],
    video_saving_queue=None,
):
    global violationReferenceId
    global frame_sending_start_time
    global frame_override_time_interval
    global unq_req_id_mq

    creds = pika.PlainCredentials(RABBIT_MQ_USERNAME, RABBIT_MQ_PASSWORD)
    connection = pika.BlockingConnection(
        pika.ConnectionParameters(
            host=RABBIT_MQ_HOST, virtual_host=RABBIT_MQ_VHOST, credentials=creds
        )
    )
    channel = connection.channel()

    # Topic Exchange changes

    # Publishing
    new_publishing_queue = f"ui.{data_json['camera_id']}_{usecase_id}"
    # logger.debug(f'Publish Queue = {new_publishing_queue}')

    channel.exchange_declare(
        exchange="topic.cctvai", exchange_type="topic", durable=True
    )

    # Subscribe
    new_subscription_queue = f"live.{data_json['camera_id']}.{usecase_id}"

    # logger.debug(f'Subscribe Queue = {new_subscription_queue}')

    channel_live = connection.channel()

    channel_live.exchange_declare(
        exchange="topic.cctvai", exchange_type="topic", durable=True
    )

    channel_live.queue_declare(
        queue="cctvai.liverequest",
        durable=True,
        arguments={"x-message-ttl": 90000},
        exclusive=False,
        auto_delete=False,
    )

    channel_live.queue_bind(
        exchange="topic.cctvai", queue="cctvai.liverequest", routing_key="live.#"
    )

    send_override_frames = False
    if not frame_override_time_interval:
        override_live_frames, frame_override_time_interval = (
            check_and_process_all_messages(
                channel_live, usecase_id, data_json["camera_id"]
            )
        )
        if override_live_frames:
            send_override_frames = True
            frame_sending_start_time = datetime.now()
    elif frame_override_time_interval and frame_sending_start_time:
        diff = (datetime.now() - frame_sending_start_time).total_seconds()
        if diff > frame_override_time_interval:
            frame_override_time_interval = None
            frame_sending_start_time = None
        else:
            send_override_frames = True

    if send_override_frames == False and unq_req_id_mq is not None:
        unq_req_id_mq = None

    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y-%H:%M:%S")
    data_json["date"] = now.strftime("%d/%m/%Y")
    data_json["time"] = now.strftime("%H:%M:%S")
    violation_dir = type_of_violation
    if "traffic_rules" in type_of_violation:
        violation_dir = "traffic_rules"
    output_filename = (
        f'{type_of_violation}_video_{data_json["camera_id"]}_{dt_string}.avi'
    )
    output_path = f"{violation_dir}/output/{output_filename}"
    output_path_mp4 = (
        f'{violation_dir}/output/{output_filename.replace(".avi", ".mp4")}'
    )

    # Is Live
    data_json["is_live"] = send_override_frames
    data_json["violation"] = violation

    # if (violation and not start_time) or (violation and start_time and (time.time() - start_time) < 140) or \
    #         (start_time and (time.time() - start_time) < 140):

    time_rec = len(output_video_1_list) / FPS if output_video_1_list else 0
    # MAX_VIDEO_THRESHOLD = 50
    # if not output_video_1_list:
    #     output_video_1_list = [frame]

    logger.debug(
        f"Condition = {violation} and {time_rec} < {MAX_VIDEO_THRESHOLD} or {send_override_frames}"
    )
    #if (violation and time_rec < MAX_VIDEO_THRESHOLD) or send_override_frames:
    if (time_rec < MAX_VIDEO_THRESHOLD) or send_override_frames:
        # # Write the both outputs video to a local folders
        if output_video_1_list is None and violation:
            violationReferenceId = datetime.now().strftime("%Y%m-%d%H-%M%S-") + str(
                uuid.uuid4()
            )
            output_video_1_list = [frame]
        else:
            if output_video_1_list is not None:
                if (time_rec < MIN_VIDEO_THRESHOLD and violation) or (
                    time_rec >= MIN_VIDEO_THRESHOLD
                ):
                    output_video_1_list.append(frame)
                else:
                    output_video_1_list = []

        data_json["type_of_violation"] = type_of_violation
        data_json["usecase_id"] = usecase_id
        data_json["violationReferenceId"] = violationReferenceId
        data_json["unqReqId"] = unq_req_id_mq

        _, buffer = cv2.imencode(".jpg", frame)
        img_base64 = base64.b64encode(buffer).decode("utf-8")

        combined_data = {"image": img_base64, "metadata": data_json}

        combined_data_bytes = json.dumps(combined_data, cls=CustomJSONEncoder).encode(
            "utf-8"
        )
        all_data.append(copy.deepcopy(data_json))
        try:
            if (time_rec >= MIN_VIDEO_THRESHOLD) or send_override_frames:
                logger.debug(f"Rabbit MQ Publish **")
                channel.basic_publish(
                    exchange="topic.cctvai",
                    routing_key=new_publishing_queue,
                    body=combined_data_bytes,
                    properties=None,
                )
        except:
            error_logger.error("RabbitMQError message not sent ")

    else:
        if output_video_1_list is not None:
            if time_rec > MIN_VIDEO_THRESHOLD:

                video_saving_queue.put(
                    {
                        "output_video_1_list": output_video_1_list,
                        "frame": frame,
                        "output_path": output_path,
                        "output_filename": output_filename,
                        "output_path_mp4": output_path_mp4,
                        "all_data": all_data,
                        "usecase_id": usecase_id,
                        "zone_id": zone_id,
                        "data_json": data_json,
                    }
                )

                # New Changes for independent process

                #current_time = datetime.now()
                #current_time = current_time.replace(tzinfo=dt.timezone.utc)
                #current_time = current_time.isoformat()
                # writer = imageio.get_writer(output_path, fps=FPS)

                #fourcc1 = cv2.VideoWriter_fourcc(*"MJPG")
                #output_video_1 = cv2.VideoWriter(
                #    output_path,
                #    fourcc1,
                #    FPS,
                #    (int(frame.shape[1]), int(frame.shape[0])),
                #    True,
                #)

                #for item in output_video_1_list:
                    # writer.append_data(item)
                #    output_video_1.write(item)
                #output_video_1.release()
                # writer.close()
                #if os.name == "posix":
                #    os.system("sync")
                #input_stream = ffmpeg.input(output_path)
                #output_stream = ffmpeg.output(
                #    input_stream, output_path_mp4, vcodec="libx264"
                #)
                #ffmpeg.run(output_stream)
                #if os.path.exists(output_path_mp4):
                #    os.remove(output_path)
                #remote_path = os.path.join(
                #    MINIO_PATH, output_filename.replace(".avi", ".mp4")
                #)
                #remote_path = remote_path.replace(
                #    os.sep, "/"
                #)  # Replace \ with / on Windows
                #try:
                #    client.fput_object(MINIO_BUCKET_NAME, remote_path, output_path_mp4)
                #    expiration_time = timedelta(seconds=604800)
                #    presigned_url = client.presigned_get_object(
                #        MINIO_BUCKET_NAME, remote_path, expiration_time
                #    )
                #    if presigned_url:
                #        all_data_json = json.dumps(all_data, cls=CustomJSONEncoder)
                #        payload = {
                #            "productId": data_json["camera_id"],
                #            "usecaseId": usecase_id,
                #            "zoneId": zone_id,
                #            "violationType": "Violation",
                #            "cameraUrl": data_json["camera_ip"],
                #            "videoUrl": remote_path,
                #            "metaViolation": all_data_json,
                #            "metaCamera": "{}",
                #            "metaZones": "{}",
                #            "recordedOn": current_time,
                #            "violationReferenceId": violationReferenceId,
                #        }

                        # video_req = requests.post(VIOLATION_FEED_CREATION_URL, json=payload, headers={'Authorization': f'Bearer {access_token}'})
                #        video_req = API(
                #            RequestMethod.POST,
                #            VIOLATION_FEED_CREATION_URL,
                #            json=payload,
                #            headers={"Authorization": f"Bearer {access_token}"},
                #        )

                #        if video_req.status_code != 200:
                #            error_logger.error(
                #                "There was some ERROR with the DB server!"
                #            )
                #        else:
                #            logger.debug(f"VideoURL updated to DB: {presigned_url}")
                #except Exception as e:
                #    error_logger.error(f"Upload Error: {output_path_mp4}")

                # current_time = datetime.now()
                # current_time = current_time.replace(tzinfo=dt.timezone.utc)
                # current_time = current_time.isoformat()

                # fourcc1 = cv2.VideoWriter_fourcc(*"MJPG")
                # output_video_1 = cv2.VideoWriter(output_path, fourcc1, FPS,
                #                                 (int(frame.shape[1]), int(frame.shape[0])), True)

                # for item in output_video_1_list:
                #    output_video_1.write(item)
                # output_video_1.release()

                # if os.name == 'posix':
                #    os.system('sync')
                # input_stream = ffmpeg.input(output_path)
                # output_stream = ffmpeg.output(input_stream, output_path_mp4, vcodec='libx264')
                # ffmpeg.run(output_stream)
                # if os.path.exists(output_path_mp4):
                #    os.remove(output_path)
                # remote_path = os.path.join(
                #    MINIO_PATH, output_filename.replace('.avi', '.mp4'))
                # remote_path = remote_path.replace(
                #    os.sep, "/")  # Replace \ with / on Windows
                # try:
                #    client.fput_object(MINIO_BUCKET_NAME, remote_path, output_path_mp4)
                #    expiration_time = timedelta(seconds=604800)
                #    presigned_url = client.presigned_get_object(MINIO_BUCKET_NAME, remote_path, expiration_time)
                #    if presigned_url:
                #        all_data_json = json.dumps(all_data, cls=CustomJSONEncoder)
                #        payload = {"productId": data_json['camera_id'],
                #                   "usecaseId": usecase_id,
                #                   "zoneId": zone_id,
                #                   "violationType": 'Violation',
                #                   "cameraUrl": data_json['camera_ip'],
                #                   "videoUrl": remote_path,
                #                   'metaViolation': all_data_json,
                #                   "metaCamera": "{}",
                #                   "metaZones": "{}",
                #                   "recordedOn": current_time,
                #                   "violationReferenceId": violationReferenceId}

                #        video_req = API(RequestMethod.POST, VIOLATION_FEED_CREATION_URL, json=payload,
                #                        headers={'Authorization': f'Bearer {access_token}'})

                #        if video_req.status_code != 200:
                #            error_logger.error("There was some ERROR with the DB server!")
                #        else:
                #            logger.debug(f"VideoURL updated to DB: {presigned_url}")
                # except Exception as e:
                #    error_logger.error(f"Upload Error: {output_path_mp4}")
                # if os.path.exists(output_path_mp4):
                #    os.remove(output_path_mp4)

                # --
                output_video_1_list = None
                start_time = None
                violationReferenceId = None
                all_data = []
                time_rec = 0
                total_time = 0
                output_video_frame_list = []

        #elif not violation and time_rec < MIN_VIDEO_THRESHOLD:
        #    time_rec = 0
        #    output_video_1_list = None
        #    start_time = None
        #    violationReferenceId = None
        #    all_data = []
        #    time_rec = 0
        #    total_time = 0
        #    output_video_frame_list = []


    return output_video_1_list, start_time, all_data, output_video_frame_list


def write_video_upload_minio_wrong_turn(
    frame,
    data_json,
    output_video_1_list,
    violation,
    type_of_violation,
    start_time,
    usecase_id,
    all_data,
    zone_id=None,
    output_video_frame_list=[],
    output_video_frames={},
    violation_video_frames={},
    video_saving_queue=None,
):
    global violationReferenceId
    global frame_sending_start_time
    global frame_override_time_interval
    global unq_req_id_mq

    creds = pika.PlainCredentials(RABBIT_MQ_USERNAME, RABBIT_MQ_PASSWORD)
    connection = pika.BlockingConnection(
        pika.ConnectionParameters(
            host=RABBIT_MQ_HOST, virtual_host=RABBIT_MQ_VHOST, credentials=creds
        )
    )
    channel = connection.channel()

    # Topic Exchange changes

    # Publishing
    new_publishing_queue = f"ui.{data_json['camera_id']}_{usecase_id}"
    # logger.debug(f'Publish Queue = {new_publishing_queue}')

    channel.exchange_declare(
        exchange="topic.cctvai", exchange_type="topic", durable=True
    )

    # Subscribe
    new_subscription_queue = f"live.{data_json['camera_id']}.{usecase_id}"

    # logger.debug(f'Subscribe Queue = {new_subscription_queue}')

    channel_live = connection.channel()

    channel_live.exchange_declare(
        exchange="topic.cctvai", exchange_type="topic", durable=True
    )

    channel_live.queue_declare(
        queue="cctvai.liverequest",
        durable=True,
        arguments={"x-message-ttl": 90000},
        exclusive=False,
        auto_delete=False,
    )

    channel_live.queue_bind(
        exchange="topic.cctvai", queue="cctvai.liverequest", routing_key="live.#"
    )

    send_override_frames = False
    if not frame_override_time_interval:
        override_live_frames, frame_override_time_interval = (
            check_and_process_all_messages(
                channel_live, usecase_id, data_json["camera_id"]
            )
        )
        if override_live_frames:
            send_override_frames = True
            frame_sending_start_time = datetime.now()
    elif frame_override_time_interval and frame_sending_start_time:
        diff = (datetime.now() - frame_sending_start_time).total_seconds()
        if diff > frame_override_time_interval:
            frame_override_time_interval = None
            frame_sending_start_time = None
        else:
            send_override_frames = True

    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y-%H:%M:%S")
    data_json["date"] = now.strftime("%d/%m/%Y")
    data_json["time"] = now.strftime("%H:%M:%S")
    violation_dir = type_of_violation
    if "traffic_rules" in type_of_violation:
        violation_dir = "traffic_rules"
    output_filename = (
        f'{type_of_violation}_video_{data_json["camera_id"]}_{dt_string}.avi'
    )
    output_path = f"{violation_dir}/output/{output_filename}"
    output_path_mp4 = (
        f'{violation_dir}/output/{output_filename.replace(".avi", ".mp4")}'
    )

    # Is Live
    data_json["is_live"] = send_override_frames
    data_json["violation"] = violation

    # if (violation and not start_time) or (violation and start_time and (time.time() - start_time) < 140) or \
    #         (start_time and (time.time() - start_time) < 140):

    time_rec = len(output_video_1_list) / FPS if output_video_1_list else 0

    # New Changes
    total_time = len(output_video_frame_list) / FPS if output_video_frame_list else 0
    if not output_video_frame_list and violation:
        output_video_frame_list.append(frame)
    else:
        if output_video_frame_list:
            if total_time <= MAX_VIDEO_THRESHOLD:
                output_video_frame_list.append(frame)
            else:
                output_video_frame_list = []

    # MAX_VIDEO_THRESHOLD = 50
    # if not output_video_1_list:
    #     output_video_1_list = [frame]

    logger.debug(
        f"Condition = {violation} and {time_rec} < {MAX_VIDEO_THRESHOLD} or {send_override_frames}"
    )
    #if (violation and time_rec < MAX_VIDEO_THRESHOLD) or send_override_frames:
    if (time_rec < MAX_VIDEO_THRESHOLD) or send_override_frames:
        # # Write the both outputs video to a local folders
        if output_video_1_list is None and violation:
            violationReferenceId = datetime.now().strftime("%Y%m-%d%H-%M%S-") + str(
                uuid.uuid4()
            )
            output_video_1_list = [frame]
        else:
            if output_video_1_list is not None:
                if (time_rec < MIN_VIDEO_THRESHOLD and violation) or (
                    time_rec >= MIN_VIDEO_THRESHOLD
                ):
                    output_video_1_list.append(frame)
                else:
                    output_video_1_list = []

        data_json["type_of_violation"] = type_of_violation
        data_json["usecase_id"] = usecase_id
        data_json["violationReferenceId"] = violationReferenceId
        data_json["unqReqId"] = unq_req_id_mq

        _, buffer = cv2.imencode(".jpg", frame)
        img_base64 = base64.b64encode(buffer).decode("utf-8")

        combined_data = {"image": img_base64, "metadata": data_json}

        combined_data_bytes = json.dumps(combined_data, cls=CustomJSONEncoder).encode(
            "utf-8"
        )
        all_data.append(copy.deepcopy(data_json))
        try:
            if (time_rec >= MIN_VIDEO_THRESHOLD) or send_override_frames:
                logger.debug(f"Rabbit MQ Publish **")
                channel.basic_publish(
                    exchange="topic.cctvai",
                    routing_key=new_publishing_queue,
                    body=combined_data_bytes,
                    properties=None,
                )
        except:
            error_logger.error("RabbitMQError message not sent ")

    if output_video_1_list is not None:
        violation_frame_copy = copy.deepcopy(violation_video_frames)
        output_video_frames_copy = copy.deepcopy(output_video_frames)

        logger.debug("New Condition!!! ")
        logger.debug(f"Violation Frames = {violation_video_frames.keys()}")
        logger.debug(f"Object Frames = {output_video_frames.keys()}")

        for key in violation_frame_copy.keys():

            if len(violation_frame_copy[key]) <= 5:
                continue

            # violation_time_rec = len(violation_video_frames[key])/FPS if violation_video_frames[key] else 0
            violation_time_rec = (
                len(output_video_frames[key]) / FPS if output_video_frames[key] else 0
            )

            logger.debug(
                f"Detection ID = {key} and {violation_time_rec} < {MAX_VIDEO_THRESHOLD}"
            )

            if (
                violation_time_rec <= MAX_VIDEO_THRESHOLD
                and violation_time_rec >= MIN_VIDEO_THRESHOLD
            ):

                if key in output_video_frames_copy:

                    video_saving_queue.put(
                        {
                            "output_video_1_list": output_video_frames[key],
                            "frame": frame,
                            "output_path": output_path,
                            "output_filename": output_filename,
                            "output_path_mp4": output_path_mp4,
                            "all_data": all_data,
                            "usecase_id": usecase_id,
                            "zone_id": zone_id,
                            "data_json": data_json,
                        }
                    )

                    # current_time = datetime.now()
                    # current_time = current_time.replace(tzinfo=dt.timezone.utc)
                    # current_time = current_time.isoformat()
                    # # writer = imageio.get_writer(output_path, fps=FPS)
                    #
                    # fourcc1 = cv2.VideoWriter_fourcc(*"MJPG")
                    # output_video_1 = cv2.VideoWriter(
                    #     output_path,
                    #     fourcc1,
                    #     FPS,
                    #     (int(frame.shape[1]), int(frame.shape[0])),
                    #     True,
                    # )
                    #
                    # video_1_list = []
                    # max_length = 0
                    #
                    # for item in output_video_frames[key]:
                    #     # writer.append_data(item)
                    #     output_video_1.write(item)
                    # output_video_1.release()
                    # # writer.close()
                    # if os.name == "posix":
                    #     os.system("sync")
                    # input_stream = ffmpeg.input(output_path)
                    # output_stream = ffmpeg.output(
                    #     input_stream, output_path_mp4, vcodec="libx264"
                    # )
                    # ffmpeg.run(output_stream)
                    # if os.path.exists(output_path_mp4):
                    #     os.remove(output_path)
                    # remote_path = os.path.join(
                    #     MINIO_PATH, output_filename.replace(".avi", ".mp4")
                    # )
                    # remote_path = remote_path.replace(
                    #     os.sep, "/"
                    # )  # Replace \ with / on Windows
                    # try:
                    #     client.fput_object(
                    #         MINIO_BUCKET_NAME, remote_path, output_path_mp4
                    #     )
                    #     expiration_time = timedelta(seconds=604800)
                    #     presigned_url = client.presigned_get_object(
                    #         MINIO_BUCKET_NAME, remote_path, expiration_time
                    #     )
                    #     if presigned_url:
                    #         all_data_json = json.dumps(all_data, cls=CustomJSONEncoder)
                    #         payload = {
                    #             "productId": data_json["camera_id"],
                    #             "usecaseId": usecase_id,
                    #             "zoneId": zone_id,
                    #             "violationType": "Violation",
                    #             "cameraUrl": data_json["camera_ip"],
                    #             "videoUrl": remote_path,
                    #             "metaViolation": all_data_json,
                    #             "metaCamera": "{}",
                    #             "metaZones": "{}",
                    #             "recordedOn": current_time,
                    #             "violationReferenceId": violationReferenceId,
                    #         }
                    #
                    #         # video_req = requests.post(VIOLATION_FEED_CREATION_URL, json=payload, headers={'Authorization': f'Bearer {access_token}'})
                    #         video_req = API(
                    #             RequestMethod.POST,
                    #             VIOLATION_FEED_CREATION_URL,
                    #             json=payload,
                    #             headers={"Authorization": f"Bearer {access_token}"},
                    #         )
                    #
                    #         if video_req.status_code != 200:
                    #             error_logger.error(
                    #                 "There was some ERROR with the DB server!"
                    #             )
                    #         else:
                    #             logger.debug(f"VideoURL updated to DB: {presigned_url}")
                    # except Exception as e:
                    #     error_logger.error(f"Upload Error: {output_path_mp4}")
                    # if os.path.exists(output_path_mp4):
                    #     os.remove(output_path_mp4)

                    output_video_1_list = None
                    start_time = None
                    violationReferenceId = None
                    all_data = []
                    time_rec = 0
                    total_time = 0
                    output_video_frame_list = []
                    output_video_frames.pop(key)
                    violation_video_frames.pop(key)

    return (
        output_video_1_list,
        start_time,
        all_data,
        output_video_frame_list,
        output_video_frames,
        violation_video_frames,
    )
