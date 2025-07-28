import os
import sys
import json
from collections import deque
from queue import Queue, SimpleQueue
import threading
from copy import deepcopy
from ultralytics import YOLO
import signal
from helpers.log_manager import logger, error_logger
from helpers.preset_manager import set_preset, get_camera_calibration
from helpers.zone_manager import fetch_zone_camera_use_case
import multiprocessing as mp
from settings import (
    UC_TRAFFIC_RULE_SPEED_LIMIT,
    access_token,
    RTSP_URL_PATTERN,
    USE_CASES_RETRIEVAL_URL,
    USECASE_ABBR,
    WIDTH_OG,
    CALIBRATION_IMG_PATH,
    HEIGHT_OG,
    SIZE_FRAME,
    RTSP_TO_PORT,
    PUB_PORT,
    CAMERA_RETRIEVAL_BY_IP_URL,
    BACKEND_URL,
)

import torch

from helpers.colors import bcolors
import numpy as np
import imutils
import time
import math
import cv2
from helpers.deepsort_manager import init_tracker
from helpers.violations_manager import write_video_upload_minio, generate_video_and_save

from scipy.spatial import distance as dist
from helpers.predict import (
    estimatespeed,
    UI_box,
    xyxy_to_xywh,
    compute_color_for_labels,
)
from helpers.bird_view_transfo_functions import (
    reverse_perspective_transform,
    compute_perspective_transform,
    compute_point_perspective_transformation,
    get_centroids_and_groundpoints,
    COLOR_GREEN,
    COLOR_RED,
    BIG_CIRCLE,
    SMALL_CIRCLE,
    COLOR_BLUE,
)
from helpers.colors import bcolors
from helpers.gpu_usage import connect_to_gpu
from common import RequestMethod, API
from violation_zeromq import process_frames
import psutil
import pynvml
import yt_dlp
from concurrent.futures import ThreadPoolExecutor
from threading import Thread
import atexit


def get_youtube_live_stream_url(video_url):
    # Use yt-dlp to retrieve the live stream URL
    ydl_opts = {"format": "best"}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(video_url, download=False)
        stream_url = info["url"]
    return stream_url


def generate_rectangle_coordinates(top_left, bottom_right):
    # Extracting coordinates
    x1, y1 = top_left
    x2, y2 = bottom_right

    # Generating top right and bottom left coordinates
    top_right = (x2, y1)
    bottom_left = (x1, y2)

    return top_right, bottom_left


def is_inside_bbox(point, bbox):
    zone_bottom_left, zone_top_left, zone_top_right, zone_bottom_right = bbox

    x = int(point[0])
    y = int(point[1])

    if cv2.pointPolygonTest(np.array(bbox, dtype=np.int32), (x, y), False) >= 0:
        return True
    return False


torch.cuda.set_device(0)


class PPEKit:
    def __init__(self, use_case, frame_queue, stop_event, video_saving_queue):
        self.classNames = None
        self.start_time = None
        self.matrix = None
        self.bird_view_img = None
        self.device = None
        self.to_be_checked = False
        self.all_data = []
        self.RTSP_URL = None
        self.subscribed_topic = None
        self.zone_id = None
        self.model_yolo = None
        self.center = None
        self.deepsort = None
        self.rotation_matrix = None
        self.threshold = None
        self.width = None
        self.height = None
        self.corner_points = None
        self.img_path = None
        self.use_case_req = None
        self.camera_data = None
        self.use_case = use_case
        self.type_of_violation = self.use_case
        self.data_deque = {}
        self.violation = False
        self.speed_line_queue = {}
        self.centroids = {}
        self.bboxes = {}
        self.speeds = {}
        self.downoid_objs = {}
        self.obj_names = {}
        self.COUNTER = 0
        self.COUNT = 0
        self.objects_zone = 3
        self.waiting_obj = {}
        self.camera_ip = ""
        self.output_video_1 = None
        self.output_video_frame_list = []
        self.zone_points = []
        yt_video_url = "https://www.youtube.com/live/6dp-bvQ7RWo?si=bE8v3svt3sKLEtfQ"
        #self.stream_url = get_youtube_live_stream_url(yt_video_url)
        self.stream_url = yt_video_url
        self.stop_event = stop_event
        self.frame_queue = frame_queue
        self.video_saving_queue = video_saving_queue
        self.setup()

    def draw_boxes(
        self,
        img,
        bbox,
        names,
        object_id,
        identities=None,
        index=None,
        offset=(0, 0),
        threshold=None,
        downoid=None,
        bird_view_image=None,
    ):
        try:
            height, width, _ = img.shape
            # remove tracked point from buffer if object is lost
            for key in list(self.data_deque):
                if key not in identities:
                    self.data_deque.pop(key)

            x1, y1, x2, y2 = [int(i) for i in bbox]
            new_center = (int((x2 + x1) / 2), int((y2 + y1) / 2))

            x1 += offset[0]
            x2 += offset[0]
            y1 += offset[1]
            y2 += offset[1]

            # code to find center of bottom edge
            center = (int((x2 + x1) / 2), int((y2 + y2) / 2))

            # get ID of object
            id = int(identities[index]) if identities is not None else 0

            # create new buffer for new object
            if id not in self.data_deque:
                self.data_deque[id] = deque(maxlen=64)
                self.speed_line_queue[id] = []

            obj_name = names[object_id[index]]
            if obj_name != "person":
                obj_name = "vehicle"
            label = "{}{:d}".format("", id) + ":" + "%s" % (obj_name)

            # add center to buffer
            self.data_deque[id].appendleft(center)
            if len(self.data_deque[id]) >= 2:
                object_speed = estimatespeed(
                    self.data_deque[id][1], self.data_deque[id][0]
                )
                self.speed_line_queue[id].append(object_speed)

            try:
                color = COLOR_GREEN
                if threshold:
                    color = COLOR_RED

                # x, y = downoid
                x, y = new_center
                cv2.circle(bird_view_image, (int(x), int(y)), BIG_CIRCLE, color, 2)
                cv2.circle(bird_view_image, (int(x), int(y)), SMALL_CIRCLE, color, -1)
                UI_box(bbox, img, label=label, color=color, line_thickness=2)

            except:
                pass

            return img

        except Exception as e:
            logger.debug(f"Error in draw_boxes function: {e}")

    def capture_frames(self):

        # Debug Video frames
        # cap = cv2.VideoCapture(self.stream_url)
        # cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        # cap.set(cv2.CAP_PROP_FPS, 25)
        # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 405)

        while not self.stop_event.is_set():

            # Debug video frames
            # ret, frame = cap.read()
            # if not ret:
            #    print("Camera frame missing!! Reloading the stream!!")
            #    cap = cv2.VideoCapture(self.stream_url)
            #    continue

            # RTSP stream
            frame = process_frames(PUB_PORT, self.subscribed_topic)
            if frame is None:
                logger.debug(f"frame is EMPTY!")
                continue

            if not self.frame_queue.full():
                self.frame_queue.put(frame)
        # cap.release()

    def setup(self):

        if len(sys.argv) < 2:
            error_logger.error("Camera IP address not provided")
            sys.exit(-1)
        self.camera_ip = sys.argv[1]

        camera_req = API(
            RequestMethod.GET,
            f"{CAMERA_RETRIEVAL_BY_IP_URL}/{self.camera_ip}",
            headers={"Authorization": f"Bearer {access_token}"},
        )

        logger.debug(f"Camera Res = {camera_req.status_code}")

        if camera_req.status_code != 200:
            error_logger.error("There was some ERROR with the server!")
            sys.exit(-1)
        self.camera_data = json.loads(camera_req.text)
        if not self.camera_data:
            error_logger.error(f"Camera with IP {self.camera_ip} not found!!")
            sys.exit(-1)

        logger.debug(f"Camera {self.camera_ip} Loaded!!")

        self.RTSP_URL = RTSP_URL_PATTERN.format(
            username=self.camera_data["username"],
            password=self.camera_data["password"],
            ipAddress=self.camera_ip,
            suffixRtspUrl=self.camera_data["urlSuffix"],
        )

        self.use_case_req = API(
            RequestMethod.POST,
            USE_CASES_RETRIEVAL_URL,
            json={
                "advancedSearch": {
                    "fields": ["id"],
                    "keyword": USECASE_ABBR[self.use_case],
                }
            },
            headers={"Authorization": f"Bearer {access_token}"},
        )

        if self.use_case_req.status_code != 200:
            error_logger.error("There was some ERROR with the server!")
            sys.exit(-1)

        usecases = self.use_case_req.json()
        usecase_data = usecases["data"][0]
        logger.debug(f"Use Case {usecase_data['id']} data retrieved!!")

        #########################################
        # Load the config for the top-down view #
        #########################################
        logger.debug(
            bcolors.WARNING
            + "[ Loading config for the bird view transformation ] "
            + bcolors.ENDC
        )

        self.img_path = CALIBRATION_IMG_PATH.format(ip=self.camera_ip.replace(".", "_"))

        self.corner_points = get_camera_calibration(self.camera_data["id"])

        if not self.corner_points:
            error_logger.error(
                f"Camera calibration points not found for camera {self.camera_ip}"
            )
            sys.exit(-1)

        width_og = WIDTH_OG
        height_og = HEIGHT_OG
        img_path = self.img_path
        size_frame = SIZE_FRAME

        logger.debug(bcolors.OKGREEN + " Done : [ Config loaded ] ..." + bcolors.ENDC)

        #########################################
        #     Compute transformation matrix             #
        #########################################
        # Compute  transformation matrix from the original frame
        image = cv2.imread(self.img_path)
        if image is None:
            image = np.zeros((WIDTH_OG, HEIGHT_OG, 3), dtype=np.uint8)

        self.matrix, imgOutput = compute_perspective_transform(
            self.corner_points, width_og, height_og, image
        )

        self.height, self.width, _ = imgOutput.shape
        blank_image = np.zeros((self.height, self.width, 3), np.uint8)
        self.height = blank_image.shape[0]
        self.width = blank_image.shape[1]
        dim = (self.width, self.height)
        self.threshold = None

        # Load the bird's eye view image
        img = cv2.imread("img/chemin_1.png")
        self.bird_view_img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

        self.deepsort = init_tracker()
        logger.debug("DeepSort tracker initiated!!")

        set_preset(self.camera_ip, self.camera_data)
        logger.debug("Default Preset activated!!")

        self.start_time = None

        self.center = (dim[0] // 2, dim[1] // 2)
        self.rotation_matrix = cv2.getRotationMatrix2D(
            self.center, self.camera_data["orientationAngle"], 1.0
        )

        self.output_video_1, self.output_video_frame_list = None, []

        temp_model_yolo = YOLO("models/yolov8x.pt")

        logger.debug("YOLO model loaded!!")

        self.zone_points, self.zone_id = fetch_zone_camera_use_case(
            self.camera_data["id"], usecase_data["id"]
        )
        if not self.zone_points:
            error_logger.error(
                f"Zone not found for the use case: {usecase_data['id']} with camera: {self.camera_data['id']}"
            )
            sys.exit(-1)
        logger.debug("Zone Points retrieved!!")

        # Convert the selected points to a NumPy array
        self.zone_points = np.array(self.zone_points, dtype=np.int32)

        self.all_data = []

        logger.debug(
            f"Stream started on camera {self.camera_data['id']} for "
            f"use case {usecase_data['id']}"
        )
        self.subscribed_topic = str(RTSP_TO_PORT.get(self.RTSP_URL))

        self.to_be_checked = True

        self.classNames = temp_model_yolo.model.names
        temp_model_yolo = None

    def detect(self):

        usecases = self.use_case_req.json()
        usecase_data = usecases["data"][0]

        width_og = WIDTH_OG
        height_og = HEIGHT_OG
        # img_path = self.img_path
        size_frame = SIZE_FRAME

        # model_yolo = self.model_yolo
        logger.debug("Loading model in Detect function")
        model_yolo = YOLO("models/yolov8x.pt")
        self.device = connect_to_gpu()
        model_yolo.to(self.device)

        logger.debug("Model Loaded in Detect!!")

        data_json = {
            "camera_id": self.camera_data["id"],
            "camera_ip": self.camera_ip,
            "zone_points": self.zone_points,
            "objects": [],
        }

        logger.debug(
            f"To be checked : {self.to_be_checked}, Stop Thread : {self.stop_event.is_set()}"
        )

        VEHICLE_CLASS_ID = [2, 3, 5, 7]

        current_obj = {}
        obj_points = []
        confidences = []
        pboxes = []

        while self.to_be_checked and not self.frame_queue.empty():
            logger.debug(f"Queue = {self.frame_queue.qsize()}")
            if not self.frame_queue.empty():
                frame = self.frame_queue.get()
                logger.debug("Got frame.... ")

                # Resize the image to the correct size
                frame = imutils.resize(frame, width=int(size_frame))

                # describe the type of font to be used
                font = cv2.FONT_HERSHEY_SIMPLEX
                text_1 = "Number of vehicles : " + str(self.COUNT)
                # insert text in teh video
                cv2.putText(
                    frame, text_1, (50, 50), font, 1, (0, 255, 255), 2, cv2.LINE_4
                )

                results = model_yolo(frame)
                self.violation = False
                list_boxes = []
                xywh_bboxs = []
                class_names = []
                confs = []
                oids = []

                for r in results:
                    boxes = r.boxes
                    for box in boxes:
                        # Bounding Box
                        x1, y1, x2, y2 = box.xyxy[0]

                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        w, h = x2 - x1, y2 - y1

                        # Confidence
                        conf = math.ceil((box.conf[0] * 100)) / 100
                        # Class Name
                        cls = int(box.cls[0])
                        if cls in [0, 1, 2, 3, 4, 5, 6, 7] and conf > 0.5:
                            b = [x1, y1, x2, y2]
                            list_boxes.append((x1, y1, w, h))
                            x_c, y_c, bbox_w, bbox_h = xyxy_to_xywh(*b)
                            xywh_obj = [x_c, y_c, bbox_w, bbox_h]
                            xywh_bboxs.append(xywh_obj)
                            confidences.append(conf)
                            oids.append(cls)
                            pboxes.append((x1, y1, w, h, cls))
                            class_names.append(self.classNames[cls])

                try:
                    xywhs = torch.Tensor(xywh_bboxs)
                    confss = torch.Tensor(confidences)

                    outputs = self.deepsort.update(
                        xywhs, confss, oids, frame, self.device
                    )

                    if len(outputs) > 0:
                        bbox_xyxy = outputs[:, :4]
                        identities = outputs[:, -2]
                        object_id = outputs[:, -1]

                        array_centroids, array_groundpoints = (
                            get_centroids_and_groundpoints(bbox_xyxy)
                        )

                        # Use the transform matrix to get the transformed coordonates
                        transformed_downoids = compute_point_perspective_transformation(
                            self.matrix, array_groundpoints
                        )
                        rotated_object_points = cv2.transform(
                            np.array(transformed_downoids).reshape(-1, 1, 2),
                            self.rotation_matrix,
                        )
                        rotated_object_points_list = [
                            (point[0][0], point[0][1])
                            for point in rotated_object_points
                        ]
                        self.COUNT = 0
                        # Iterate over the transformed points
                        for index, downoid in enumerate(rotated_object_points_list):

                            x1, y1, x2, y2 = [int(i) for i in bbox_xyxy[index]]

                            # New Center for bird view change
                            new_center = (int((x2 + x1) / 2), int((y2 + y1) / 2))

                            top_left = (x1, y1)
                            bottom_right = (x2, y2)

                            top_right, bottom_left = generate_rectangle_coordinates(
                                top_left, bottom_right
                            )

                            # if cv2.pointPolygonTest(zone_points, tuple(downoid), False) >= 0:

                            # Point is inside the zone
                            #    if not (downoid[0] > width or downoid[0] < 0 or downoid[1] > height + 200 or downoid[1] < 0):
                            #        try:
                            #            if list_boxes[index][0]:
                            #                COUNT += 1
                            #        except Exception as e:
                            #            continue
                            #        if COUNT >= objects_zone:
                            #            # write the scripts for sending violation
                            #            violation = True
                            #            if not start_time:
                            #                start_time = time.time()
                            #            color = COLOR_RED
                            #            # Iterate over the transformed points
                            #            for index, downoid in enumerate(transformed_downoids):

                            #                if cv2.pointPolygonTest(zone_points, tuple(downoid), False) >= 0:

                            #                    # Point is inside the zone
                            #                    if not (downoid[0] > width or downoid[0] < 0 or downoid[1] > height + 200 or
                            #                            downoid[
                            #                                1] < 0):
                            #                        draw_boxes(frame, bbox_xyxy[index], classNames, object_id, identities,
                            #                                   index=index, threshold=True,
                            #                                   downoid=downoid, bird_view_image=bird_view_img)
                            #        else:
                            #            color = COLOR_GREEN
                            #            draw_boxes(frame, bbox_xyxy[index], classNames, object_id, identities,
                            #                       index=index, downoid=downoid, bird_view_image=bird_view_img)
                            #        current_object_point = downoid[0], downoid[1]
                            #        if current_object_point not in obj_points:
                            #            obj_points.append(current_object_point)
                            #            current_obj['object_point'] = current_object_point
                            #            current_obj['object_id'] = int(
                            #                identities[index]) if identities is not None else 0
                            #            try:
                            #                current_obj['class'] = class_names[index]
                            #                current_obj['state'] = 'unsafe' if color == COLOR_RED else 'safe'
                            #            except Exception as e:
                            #                continue

                            #            data_json['objects'].append(deepcopy(current_obj))

                            if (
                                is_inside_bbox(bottom_left, self.zone_points)
                                or is_inside_bbox(top_left, self.zone_points)
                                or is_inside_bbox(top_right, self.zone_points)
                                or is_inside_bbox(bottom_right, self.zone_points)
                            ):

                                # Point is inside the zone

                                try:
                                    if list_boxes[index][0]:
                                        self.COUNT += 1
                                except Exception as e:
                                    continue
                                if self.COUNT >= self.objects_zone:
                                    # write the scripts for sending violation
                                    self.violation = True
                                    if not self.start_time:
                                        self.start_time = time.time()
                                    color = COLOR_RED
                                    # Iterate over the transformed points
                                    for index, downoid in enumerate(
                                        transformed_downoids
                                    ):

                                        if (
                                            is_inside_bbox(
                                                bottom_left, self.zone_points
                                            )
                                            or is_inside_bbox(
                                                top_left, self.zone_points
                                            )
                                            or is_inside_bbox(
                                                top_right, self.zone_points
                                            )
                                            or is_inside_bbox(
                                                bottom_right, self.zone_points
                                            )
                                        ):

                                            # Point is inside the zone

                                            self.draw_boxes(
                                                frame,
                                                bbox_xyxy[index],
                                                self.classNames,
                                                object_id,
                                                identities,
                                                index=index,
                                                threshold=True,
                                                downoid=downoid,
                                                bird_view_image=self.bird_view_img,
                                            )
                                else:
                                    color = COLOR_GREEN
                                    self.draw_boxes(
                                        frame,
                                        bbox_xyxy[index],
                                        self.classNames,
                                        object_id,
                                        identities,
                                        index=index,
                                        downoid=downoid,
                                        bird_view_image=self.bird_view_img,
                                    )

                                # current_object_point = downoid[0], downoid[1]
                                current_object_point = new_center[0], new_center[1]
                                if current_object_point not in obj_points:
                                    obj_points.append(current_object_point)
                                    current_obj["object_point"] = current_object_point
                                    current_obj["object_id"] = (
                                        int(identities[index])
                                        if identities is not None
                                        else 0
                                    )
                                    try:
                                        current_obj["class"] = class_names[index]
                                        current_obj["state"] = (
                                            "unsafe" if color == COLOR_RED else "safe"
                                        )
                                    except Exception as e:
                                        continue

                                    data_json["objects"].append(deepcopy(current_obj))

                except Exception as e:
                    logger.debug(f"Error in Detect loop {e}")

                # # Show both images
                if os.environ.get("DEBUG") == "1":
                    cv2.imshow("Bird view", self.bird_view_img)
                    cv2.imshow("Video", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

                logger.debug(f"Calling Rebbit MQ")
                # logger.debug(f"Video list = {self.output_video_1}")

                (
                    self.output_video_1,
                    self.start_time,
                    self.all_data,
                    self.output_video_frame_list,
                ) = write_video_upload_minio(
                    frame,
                    data_json,
                    self.output_video_1,
                    self.violation,
                    self.use_case,
                    self.start_time,
                    usecase_data["id"],
                    self.all_data,
                    zone_id=self.zone_id,
                    output_video_frame_list=self.output_video_frame_list,
                    video_saving_queue=self.video_saving_queue,
                )
                # time.sleep(0.5)

            else:
                logger.debug("Waiting for frames.... ")
                time.sleep(0.01)


def remove_lock_file(lock_file_path):
    logger.debug(f"In remove Lock file {lock_file_path}.")
    if os.path.exists(lock_file_path):
        os.remove(lock_file_path)
        logger.debug(f"Lock file {lock_file_path} removed.")


def cleanup(
    capture_process,
    detect_process,
    stop_event,
    lock_file_path,
    video_saving_queue,
    video_saving_process,
):
    """Clean up resources on exit."""
    try:
        logger.debug("Stopping processes and cleaning up resources...")

        stop_event.set()

        if capture_process.is_alive():
            capture_process.terminate()
            logger.debug("Capture process terminated.")
        if detect_process.is_alive():
            detect_process.terminate()
            logger.debug("Detection process terminated.")
        if video_saving_process.is_alive():
            video_saving_process.terminate()
            logger.debug("Video saving process terminated.")

        capture_process.join()
        detect_process.join()
        video_saving_process.join()

        # Remove lock file
        remove_lock_file(lock_file_path)

        logger.debug("Cleanup completed.")
    except Exception as e:
        # Remove lock file
        remove_lock_file(lock_file_path)
        logger.debug(f"Exception in cleanup = {e}")


def get_lockfile_data(ip, usecase):

    logger.debug(f"** In Lockfile Data **")

    camera_req = API(
        RequestMethod.GET,
        f"{CAMERA_RETRIEVAL_BY_IP_URL}/{ip}",
        headers={"Authorization": f"Bearer {access_token}"},
    )

    logger.debug(f"Camera Res = {camera_req.status_code}")

    if camera_req.status_code != 200:
        error_logger.error("There was some ERROR with the server!")
        sys.exit(-1)
    camera_data = json.loads(camera_req.text)
    if not camera_data:
        error_logger.error(f"Camera with IP {ip} not found!!")
        sys.exit(-1)

    use_case_req = API(
        RequestMethod.POST,
        USE_CASES_RETRIEVAL_URL,
        json={
            "advancedSearch": {
                "fields": ["id"],
                "keyword": USECASE_ABBR[usecase],
            }
        },
        headers={"Authorization": f"Bearer {access_token}"},
    )

    if use_case_req.status_code != 200:
        error_logger.error("There was some ERROR with the server!")
        sys.exit(-1)

    usecases = use_case_req.json()
    usecase_data = usecases["data"][0]
    logger.debug(f"Use Case {usecase_data['id']} data retrieved!!")

    return f"{camera_data['id']}_{usecase_data['id']}"


def main(use_case):

    if len(sys.argv) < 2:
        error_logger.error("Camera IP address not provided")
        sys.exit(-1)
    ip = sys.argv[1]

    lock_file_name = get_lockfile_data(ip, use_case)

    logger.debug(f"Lock file = {lock_file_name}")

    # Prevent multiple instances using a lock file
    lock_file_path = f"/tmp/{lock_file_name}.lock"
    if os.path.exists(lock_file_path):
        logger.debug("Another instance of this script is already running.")
        #sys.exit(1)
    with open(lock_file_path, "w") as lock_file:
        lock_file.write(str(os.getpid()))

    # Ensure lock file is removed on exit
    atexit.register(remove_lock_file, lock_file_path)

    logger.debug("Calling main function")
    mp.set_start_method("spawn", force=True)

    frame_queue = mp.Queue(maxsize=10000)
    stop_event = mp.Event()
    video_saving_queue = mp.Queue(maxsize=1000)

    detection = PPEKit(use_case, frame_queue, stop_event, video_saving_queue)

    # Define processes for frame capture and detection
    capture_process = mp.Process(target=detection.capture_frames)
    detect_process = mp.Process(target=detection.detect)
    video_saving_process = mp.Process(
        target=generate_video_and_save, args=(video_saving_queue, stop_event)
    )

    # Register signal handlers
    def signal_handler(signum, frame):
        logger.debug(f"Received signal {signum}. Initiating cleanup.")
        cleanup(
            capture_process,
            detect_process,
            stop_event,
            lock_file_path,
            video_saving_queue,
            video_saving_process,
        )
        sys.exit(0)

    # Signal Register
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    # Start processes
    try:
        logger.debug("Starting processes...")
        capture_process.start()
        detect_process.start()
        video_saving_process.start()

        while not stop_event.is_set():
            pass
            # time.sleep(0.1)
    except Exception as e:
        logger.debug(f"Error occurred in Main: {str(e)}")
    finally:
        logger.debug(f"In finally block cleanup function in Main")
        cleanup(
            capture_process,
            detect_process,
            stop_event,
            lock_file_path,
            video_saving_queue,
            video_saving_process,
        )


if __name__ == "__main__":
    type_of_violation = "four_vehicles_per_crane"
    main(type_of_violation)
    #
    # Detection = cls_Zone(type_of_violation)
    # Detection.start()

    # python thread_test.py 192.168.141.226
