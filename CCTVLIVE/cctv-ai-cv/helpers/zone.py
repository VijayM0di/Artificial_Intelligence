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
)
from helpers.colors import bcolors
from helpers.gpu_usage import connect_to_gpu
from common import RequestMethod, API
import psutil
import pynvml
import yt_dlp
from concurrent.futures import ThreadPoolExecutor
from threading import Thread
from violation_zeromq import process_frames
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


class cls_Zone:
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
        self.waiting_obj = {}
        self.camera_ip = ""
        self.output_video_1 = None
        self.output_video_frame_list = []
        self.zone_points = []
        yt_video_url = "https://www.youtube.com/live/6dp-bvQ7RWo?si=qKjMSgVSfF_ebWRQ"
        self.stream_url = get_youtube_live_stream_url(yt_video_url)
        #self.stream_url = yt_video_url
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
        downoid=None,
        bird_view_image=None,
        threshold=None,
        color=None,
    ):

        if self.type_of_violation in [
            "people_close_to_moving_objects",
            "people_under_suspended_load",
        ]:
            pass

        height, width, _ = img.shape
        # remove tracked point from buffer if object is lost
        for key in list(self.data_deque):
            if key not in identities:
                self.data_deque.pop(key)
                self.waiting_obj.pop(key)
                if self.type_of_violation in [
                    "people_close_to_moving_objects",
                    "people_under_suspended_load",
                ]:
                    if key in self.centroids:
                        self.centroids.pop(key)
                    if key in self.bboxes:
                        self.bboxes.pop(key)

        x1, y1, x2, y2 = [int(i) for i in bbox]
        new_center = (int((x2 + x1) / 2), int((y2 + y1) / 2))
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]

        center = 0
        centroid = 0
        if (
            self.type_of_violation in ["zone_intrusion", "water_edge", "waiting_area"]
            or "traffic_rules" in self.type_of_violation
        ):
            # code to find center of bottom edge
            center = (int((x2 + x1) / 2), int((y2 + y2) / 2))
        elif self.type_of_violation in [
            "people_close_to_moving_objects",
            "people_under_suspended_load",
        ]:
            centroid = (int((x2 + x1) / 2), int((y2 + y2) / 2))

        # get ID of object
        id = int(identities[index]) if identities is not None else 0

        # create new buffer for new object
        if id not in self.data_deque:
            self.data_deque[id] = deque(maxlen=64)
            self.speed_line_queue[id] = []
            self.waiting_obj[id] = {"time": 0, "speed": [], "violation": False}

        if self.type_of_violation in ["zone_intrusion", "water_edge"]:
            color = color or compute_color_for_labels(object_id[index])
        elif (
            self.type_of_violation == "traffic_rules_illegal_parking"
            or self.type_of_violation == "waiting_area"
        ):
            color = compute_color_for_labels(object_id[index])

        obj_name = names[object_id[index]]
        if obj_name != "person":
            obj_name = "vehicle"
        label = "{}{:d}".format("", id) + ":" + "%s" % (obj_name)

        if (
            self.type_of_violation in ["zone_intrusion", "water_edge", "waiting_area"]
            or "traffic_rules" in self.type_of_violation
        ):
            # add center to buffer
            self.data_deque[id].appendleft(center)
        elif self.type_of_violation in [
            "people_close_to_moving_objects",
            "people_under_suspended_load",
            "people_under_suspended_load",
        ]:
            self.data_deque[id].appendleft(centroid)

        if len(self.data_deque[id]) >= 2:
            object_speed = estimatespeed(self.data_deque[id][1], self.data_deque[id][0])
            self.speed_line_queue[id].append(object_speed)

        start_time = 0
        if self.type_of_violation in [
            "zone_intrusion",
            "water_edge",
            "traffic_rules_no_entry",
        ]:
            # draw trail
            if color == COLOR_RED:
                self.violation = True
            for i in range(1, len(self.data_deque[id])):
                # check if on buffer value is none
                if self.data_deque[id][i - 1] is None or self.data_deque[id][i] is None:
                    continue
                if not start_time:
                    start_time = time.time()

                # x, y = downoid
                x, y = new_center
                if self.type_of_violation in ["zone_intrusion", "water_edge"]:
                    cv2.circle(bird_view_image, (int(x), int(y)), BIG_CIRCLE, color, 2)
                    cv2.circle(
                        bird_view_image, (int(x), int(y)), SMALL_CIRCLE, color, -1
                    )
                    UI_box(bbox, img, label=label, color=color, line_thickness=2)
                elif self.type_of_violation == "traffic_rules_no_entry":
                    cv2.circle(
                        bird_view_image, (int(x), int(y)), BIG_CIRCLE, COLOR_RED, 2
                    )
                    cv2.circle(
                        bird_view_image, (int(x), int(y)), SMALL_CIRCLE, COLOR_RED, -1
                    )
                    UI_box(bbox, img, label=label, color=COLOR_RED, line_thickness=2)
                    self.violation = True

        elif self.type_of_violation == "traffic_rules_illegal_parking":
            try:
                avg_speed = sum(self.speed_line_queue[id]) // len(
                    self.speed_line_queue[id]
                )
                object_speed = estimatespeed(
                    self.data_deque[id][1], self.data_deque[id][0]
                )
                logger.debug(f"**** Object Speed of {id} : {object_speed} ****")

                if 0 <= abs(object_speed) <= 2:
                    self.waiting_obj[id]["violation"] = True
                else:
                    self.waiting_obj[id]["violation"] = False

                if self.waiting_obj[id]["violation"] == True:
                    self.waiting_obj[id]["speed"].append(object_speed)
                    self.waiting_obj[id]["time"] = (
                        len(self.waiting_obj[id]["speed"]) / 20
                    )
                else:
                    self.waiting_obj[id]["speed"] = []
                    self.waiting_obj[id]["time"] = 0
                logger.debug(
                    f"**** Object Waiting  of {id} : {self.waiting_obj[id]} ****"
                )

                # if 0 <= abs(object_speed) <= 4:
                if self.waiting_obj[id]["time"] > 2:
                    self.violation = True
                    if not start_time:
                        start_time = time.time()
                    # x, y = downoid
                    x, y = new_center
                    cv2.circle(
                        bird_view_image, (int(x), int(y)), BIG_CIRCLE, COLOR_RED, 2
                    )
                    cv2.circle(
                        bird_view_image, (int(x), int(y)), SMALL_CIRCLE, COLOR_RED, -1
                    )
                    UI_box(bbox, img, label=label, color=COLOR_RED, line_thickness=2)

                    # draw trail
                    for i in range(1, len(self.data_deque[id])):
                        # check if on buffer value is none
                        if (
                            self.data_deque[id][i - 1] is None
                            or self.data_deque[id][i] is None
                        ):
                            continue

                        # generate dynamic thickness of trails
                        thickness = int(np.sqrt(64 / float(i + i)) * 1.5)
                        # draw trails
                        cv2.line(
                            img,
                            self.data_deque[id][i - 1],
                            self.data_deque[id][i],
                            color,
                            thickness,
                        )
            except Exception as e:
                pass

        elif self.type_of_violation == "waiting_area":
            try:
                avg_speed = sum(self.speed_line_queue[id]) // len(
                    self.speed_line_queue[id]
                )
                object_speed = estimatespeed(
                    self.data_deque[id][1], self.data_deque[id][0]
                )
                logger.debug(f"**** Object Speed of {id} : {object_speed} ****")

                if 0 <= abs(object_speed) <= 2:
                    self.waiting_obj[id]["violation"] = True
                else:
                    self.waiting_obj[id]["violation"] = False

                if self.waiting_obj[id]["violation"] == True:
                    self.waiting_obj[id]["speed"].append(object_speed)
                    self.waiting_obj[id]["time"] = (
                        len(self.waiting_obj[id]["speed"]) / 20
                    )
                else:
                    self.waiting_obj[id]["speed"] = []
                    self.waiting_obj[id]["time"] = 0
                logger.debug(
                    f"**** Object Waiting  of {id} : {self.waiting_obj[id]} ****"
                )

                # Detect wrong U-turns based on the change in direction
                # if 0 <= abs(object_speed) <= 4:
                if self.waiting_obj[id]["time"] > 2:
                    self.violation = True
                    if not start_time:
                        start_time = time.time()
                    # x, y = downoid
                    x, y = new_center
                    cv2.circle(
                        bird_view_image, (int(x), int(y)), BIG_CIRCLE, COLOR_RED, 2
                    )
                    cv2.circle(
                        bird_view_image, (int(x), int(y)), SMALL_CIRCLE, COLOR_RED, -1
                    )
                    UI_box(bbox, img, label=label, color=COLOR_RED, line_thickness=2)

                    # draw trail
                    for i in range(1, len(self.data_deque[id])):
                        # check if on buffer value is none
                        if (
                            self.data_deque[id][i - 1] is None
                            or self.data_deque[id][i] is None
                        ):
                            continue

                        # generate dynamic thickness of trails
                        thickness = int(np.sqrt(64 / float(i + i)) * 1.5)
                        # draw trails
                        cv2.line(
                            img,
                            self.data_deque[id][i - 1],
                            self.data_deque[id][i],
                            color,
                            thickness,
                        )
            except Exception as e:
                pass

        elif self.type_of_violation == "people_close_to_moving_objects":
            try:
                avg_speed = sum(self.speed_line_queue[id]) // len(
                    self.speed_line_queue[id]
                )
                if avg_speed <= 3:
                    return img
            except:
                pass
            try:
                avg_speed = sum(self.speed_line_queue[id]) // len(
                    self.speed_line_queue[id]
                )
                self.speeds[id] = avg_speed
                self.centroids[id] = centroid
                self.obj_names[id] = obj_name
                self.downoid_objs[id] = downoid
                self.bboxes[id] = bbox
            except:
                pass

            for i, centroid1 in self.centroids.items():
                downoid1 = self.downoid_objs[i]
                radius = 5
                for j, centroid2 in self.centroids.items():
                    downoid2 = self.downoid_objs[j]
                    if self.obj_names[i] != self.obj_names[j] and (
                        self.obj_names[i] == "person" or self.obj_names[j] == "person"
                    ):
                        if (self.obj_names[i] == "person" and self.speeds[j] > 7) or (
                            self.obj_names[j] == "person" and self.speeds[i] > 7
                        ):
                            distance = dist.euclidean(centroid1, centroid2)
                            if distance < threshold and distance > 10:
                                if (i == id and self.obj_names[i] == "person") or (
                                    j == id and self.obj_names[j] == "person"
                                ):
                                    self.violation = True
                                    if not start_time:
                                        start_time = time.time()
                                color = COLOR_RED
                                cv2.line(
                                    bird_view_image,
                                    tuple(np.int0(downoid1)),
                                    tuple(np.int0(downoid2)),
                                    color,
                                    2,
                                )
                                cv2.circle(
                                    bird_view_image,
                                    tuple(np.int0(downoid1)),
                                    radius,
                                    color,
                                    -1,
                                )
                                print(
                                    "Social distancing violation between faces ",
                                    i,
                                    " and ",
                                    j,
                                )
                                cv2.line(
                                    img,
                                    tuple(np.int0(centroid1)),
                                    tuple(np.int0(centroid2)),
                                    color,
                                    2,
                                )
                                cv2.circle(
                                    img, tuple(np.int0(centroid1)), radius, color, -1
                                )
            color = COLOR_GREEN
            if self.violation:
                color = COLOR_RED
            UI_box(bbox, img, label=label, color=color, line_thickness=2)
            # x, y = downoid
            x, y = new_center
            cv2.circle(bird_view_image, (int(x), int(y)), BIG_CIRCLE, color, 2)
            cv2.circle(bird_view_image, (int(x), int(y)), SMALL_CIRCLE, color, -1)

        elif self.type_of_violation == "people_under_suspended_load":
            try:
                avg_speed = sum(self.speed_line_queue[id]) // len(
                    self.speed_line_queue[id]
                )
                self.speeds[id] = avg_speed
                self.centroids[id] = centroid
                self.obj_names[id] = obj_name
                self.downoid_objs[id] = downoid
                # label = label + " " + str(avg_speed) + "km/h"
            except:
                pass

            radius = 5
            for i, centroid1 in self.centroids.items():
                downoid1 = self.downoid_objs[i]
                for j, centroid2 in self.centroids.items():
                    downoid2 = self.downoid_objs[j]
                    if self.obj_names[i] != self.obj_names[j] and (
                        self.obj_names[i] == "person" or self.obj_names[j] == "person"
                    ):
                        distance = dist.euclidean(centroid1, centroid2)
                        color = COLOR_RED
                        if distance < threshold:
                            self.violation = True
                            if not start_time:
                                start_time = time.time()
                            cv2.line(
                                bird_view_image,
                                tuple(np.int0(downoid1)),
                                tuple(np.int0(downoid2)),
                                color,
                                2,
                            )
                            cv2.circle(
                                bird_view_image,
                                tuple(np.int0(downoid1)),
                                radius,
                                color,
                                -1,
                            )
                            cv2.line(
                                img,
                                tuple(np.int0(centroid1)),
                                tuple(np.int0(centroid2)),
                                color,
                                2,
                            )

                            cv2.circle(
                                img, tuple(np.int0(centroid1)), radius, (0, 0, 255), -1
                            )
            color = COLOR_GREEN
            if self.violation:
                color = COLOR_RED
            UI_box(bbox, img, label=label, color=color, line_thickness=2)
            # x, y = downoid
            x, y = new_center
            cv2.circle(bird_view_image, (int(x), int(y)), BIG_CIRCLE, color, 2)
            cv2.circle(bird_view_image, (int(x), int(y)), SMALL_CIRCLE, color, -1)

        return img

    def capture_frames(self):

        # Debug Video frames
        cap = cv2.VideoCapture(self.stream_url)
        # cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        # cap.set(cv2.CAP_PROP_FPS, 25)
        # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 405)

        while not self.stop_event.is_set():

            # Debug video frames
            ret, frame = cap.read()
            if not ret:
                print("Camera frame missing!! Reloading the stream!!")
                cap = cv2.VideoCapture(self.stream_url)
                continue

            # RTSP stream
            #frame = process_frames(PUB_PORT, self.subscribed_topic)
            #if frame is None:
            #    logger.debug(f'frame is EMPTY!')
            #    continue

            if not self.frame_queue.full():
                self.frame_queue.put(frame)
        # Debug
        cap.release()

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

        logger.debug(bcolors.OKGREEN + " Done : [ Config loaded ] ..." + bcolors.ENDC)

        width_og = WIDTH_OG
        height_og = HEIGHT_OG
        img_path = self.img_path
        size_frame = SIZE_FRAME

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

        self.threshold = 0
        if self.use_case in ["zone_intrusion", "water_edge", "waiting_area"]:
            self.threshold = None
        elif (
            self.use_case
            in ["people_close_to_moving_objects", "people_under_suspended_load"]
            or "traffic_rules" in self.use_case
        ):
            self.threshold = 100

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

        temp_model_yolo = None
        if self.use_case == "people_under_suspended_load":
            temp_model_yolo = YOLO("models/container_person.pt")
        else:
            temp_model_yolo = YOLO("models/yolov8x.pt")

        logger.debug("YOLO model loaded!!")

        if (
            self.use_case in ["zone_intrusion", "water_edge", "waiting_area"]
            or "traffic_rules" in self.use_case
        ):
            self.zone_points, self.zone_id = fetch_zone_camera_use_case(
                self.camera_data["id"], usecase_data["id"]
            )
            if not self.zone_points:
                error_logger.error(
                    f"Zone not found for the use case: {usecase_data['id']} with camera: {self.camera_data['id']}"
                )
                sys.exit(-1)
            logger.debug(f"Zone Points {self.zone_points}")
            logger.debug("Zone Points retrieved!!")
        elif self.use_case in [
            "people_close_to_moving_objects",
            "people_under_suspended_load",
        ]:
            self.zone_points = []
            self.zone_id = None

        self.all_data = []

        # Loop until the end of the video stream
        self.to_be_checked = False
        frame_filename = ""
        logger.debug(
            f"Stream started on camera {self.camera_data['id']} for "
            f"use case {usecase_data['id']}"
        )
        self.subscribed_topic = str(RTSP_TO_PORT.get(self.RTSP_URL))

        if (
            self.use_case
            in [
                "zone_intrusion",
                "waiting_area",
                "water_edge",
                "people_close_to_moving_objects",
                "people_under_suspended_load",
            ]
            or "traffic_rules" in self.use_case
        ):
            self.to_be_checked = True

        # self.device = connect_to_gpu()
        # self.model_yolo.to(self.device)

        # dict maping class_id to class_name
        self.classNames = temp_model_yolo.model.names
        temp_model_yolo = None

    def detect(self):

        # type_of_violation = self.use_case
        # COUNT = self.COUNT
        # COUNTER = self.COUNTER

        # camera_ip = self.camera_ip
        # camera_data = self.camera_data

        usecases = self.use_case_req.json()
        usecase_data = usecases["data"][0]

        width_og = WIDTH_OG
        height_og = HEIGHT_OG
        # img_path = self.img_path
        size_frame = SIZE_FRAME

        # corner_points = self.corner_points

        # height, width = self.height, self.width

        # threshold = self.threshold

        # center = self.center
        # rotation_matrix = self.rotation_matrix

        # model_yolo = self.model_yolo
        logger.debug("Loading model in Detect function")
        model_yolo = YOLO("models/yolov8x.pt")
        self.device = connect_to_gpu()
        model_yolo.to(self.device)

        logger.debug("Model Loaded in Detect!!")

        # self.classNames = model_yolo.model.names

        # zone_points = self.zone_points

        data_json = {
            "camera_id": self.camera_data["id"],
            "camera_ip": self.camera_ip,
            "zone_points": self.zone_points,
        }

        # Convert the selected points to a NumPy array
        self.zone_points = np.array(self.zone_points, dtype=np.int32)

        VEHICLE_CLASS_ID = [2, 3, 5, 7]
        PERSON_CLASS_ID = 0

        # dict maping class_id to class_name
        # classNames = self.model_yolo.model.names

        logger.debug("Existing frame data cleared!!")

        # function removed

        # all_data = self.all_data

        logger.debug(
            f"To be checked : {self.to_be_checked}, Stop Thread : {self.stop_event.is_set()}"
        )

        while self.to_be_checked and not self.frame_queue.empty():
            logger.debug(f"Queue = {self.frame_queue.qsize()} ")
            if not self.frame_queue.empty():
                frame = self.frame_queue.get()
                logger.debug("Got frame.... ")

                data_json["objects"] = []
                current_obj = {}
                obj_points = []

                confidences = []
                pboxes = []

                # Load the frame
                statement_to_check = False

                # Resize the image to the correct size
                frame = imutils.resize(frame, width=int(size_frame))

                if self.use_case == "waiting_area":
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

                        if "traffic_rules" in self.use_case:
                            if cls not in VEHICLE_CLASS_ID:
                                continue

                        elif self.use_case in ["zone_intrusion"]:
                            if cls not in VEHICLE_CLASS_ID and cls != PERSON_CLASS_ID:
                                continue

                        elif self.use_case in ["people_close_to_moving_objects"]:
                            if cls not in VEHICLE_CLASS_ID and cls != PERSON_CLASS_ID:
                                continue

                        elif self.use_case in ["waiting_area"]:
                            if cls in [0, 1, 2, 3, 4, 5, 6, 7]:
                                if conf > 0.5:
                                    b = [x1, y1, x2, y2]
                                    list_boxes.append((x1, y1, w, h))
                                    x_c, y_c, bbox_w, bbox_h = xyxy_to_xywh(*b)
                                    xywh_obj = [x_c, y_c, bbox_w, bbox_h]
                                    xywh_bboxs.append(xywh_obj)
                                    confidences.append(conf)
                                    oids.append(cls)
                                    pboxes.append((x1, y1, w, h, cls))
                                    class_names.append(self.classNames[cls])
                                    self.COUNTER = len(list_boxes)
                        if self.use_case not in ["waiting_area"]:
                            currentClass = self.classNames[cls]
                            if conf > 0.5:
                                b = [x1, y1, x2, y2]
                                list_boxes.append((x1, y1, w, h))
                                x_c, y_c, bbox_w, bbox_h = xyxy_to_xywh(*b)
                                xywh_obj = [x_c, y_c, bbox_w, bbox_h]
                                xywh_bboxs.append(xywh_obj)
                                confidences.append(conf)
                                oids.append(cls)
                                class_names.append(self.classNames[cls])
                                pboxes.append((x1, y1, w, h, cls))

                try:
                    xywhs = torch.Tensor(xywh_bboxs)
                    confss = torch.Tensor(confidences)

                    outputs = self.deepsort.update(
                        xywhs, confss, oids, frame, self.device
                    )

                    if (
                        self.use_case in ["traffic_rules_illegal_parking"]
                        and len(xywh_bboxs) == 0
                    ):
                        continue

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

                        if self.use_case in ["zone_intrusion", "water_edge"]:
                            # Iterate over the transformed points
                            for index, downoid in enumerate(rotated_object_points_list):

                                # New Changes
                                x1, y1, x2, y2 = [int(i) for i in bbox_xyxy[index]]

                                # New Center for bird view change
                                new_center = (int((x2 + x1) / 2), int((y2 + y1) / 2))

                                top_left = (x1, y1)
                                bottom_right = (x2, y2)

                                top_right, bottom_left = generate_rectangle_coordinates(
                                    top_left, bottom_right
                                )

                                if (
                                    is_inside_bbox(bottom_left, self.zone_points)
                                    or is_inside_bbox(top_left, self.zone_points)
                                    or is_inside_bbox(top_right, self.zone_points)
                                    or is_inside_bbox(bottom_right, self.zone_points)
                                ):
                                    # Point is inside the zone

                                    x, y = downoid
                                    self.draw_boxes(
                                        frame,
                                        bbox_xyxy[index],
                                        self.classNames,
                                        object_id,
                                        identities,
                                        index=index,
                                        downoid=downoid,
                                        bird_view_image=self.bird_view_img,
                                        color=COLOR_RED,
                                    )

                                    # current_object_point = downoid[0], downoid[1]
                                    current_object_point = new_center[0], new_center[1]
                                    if current_object_point not in obj_points:
                                        obj_points.append(current_object_point)
                                        current_obj["object_point"] = (
                                            current_object_point
                                        )
                                        current_obj["class"] = self.classNames[
                                            object_id[index]
                                        ]
                                        current_obj["state"] = "unsafe"
                                        current_obj["object_id"] = (
                                            int(identities[index])
                                            if identities is not None
                                            else 0
                                        )
                                        data_json["objects"].append(
                                            deepcopy(current_obj)
                                        )

                                else:
                                    self.draw_boxes(
                                        frame,
                                        bbox_xyxy[index],
                                        self.classNames,
                                        object_id,
                                        identities,
                                        index=index,
                                        downoid=downoid,
                                        bird_view_image=self.bird_view_img,
                                        color=COLOR_GREEN,
                                    )

                                    # current_object_point = downoid[0], downoid[1]
                                    current_object_point = new_center[0], new_center[1]
                                    if current_object_point not in obj_points:
                                        obj_points.append(current_object_point)
                                        current_obj["object_point"] = (
                                            current_object_point
                                        )
                                        current_obj["class"] = self.classNames[
                                            object_id[index]
                                        ]
                                        current_obj["state"] = "safe"
                                        current_obj["object_id"] = (
                                            int(identities[index])
                                            if identities is not None
                                            else 0
                                        )
                                        data_json["objects"].append(
                                            deepcopy(current_obj)
                                        )

                        elif "traffic_rules" in self.use_case:
                            # Iterate over the transformed points
                            for index, downoid in enumerate(rotated_object_points_list):
                                self.violation = False
                                color = COLOR_GREEN

                                # New Changes
                                x1, y1, x2, y2 = [int(i) for i in bbox_xyxy[index]]

                                # New Center for bird view change
                                new_center = (int((x2 + x1) / 2), int((y2 + y1) / 2))

                                top_left = (x1, y1)
                                bottom_right = (x2, y2)

                                top_right, bottom_left = generate_rectangle_coordinates(
                                    top_left, bottom_right
                                )

                                if (
                                    is_inside_bbox(bottom_left, self.zone_points)
                                    or is_inside_bbox(top_left, self.zone_points)
                                    or is_inside_bbox(top_right, self.zone_points)
                                    or is_inside_bbox(bottom_right, self.zone_points)
                                ):
                                    # Point is inside the zone

                                    x, y = downoid
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
                                            "unsafe" if self.violation else "safe"
                                        )
                                    except Exception as e:
                                        continue
                                    data_json["objects"].append(deepcopy(current_obj))
                        elif self.use_case in [
                            "people_close_to_moving_objects",
                            "people_under_suspended_load",
                        ]:
                            for index, downoid in enumerate(rotated_object_points_list):
                                self.violation = False

                                # New Changes
                                x1, y1, x2, y2 = [int(i) for i in bbox_xyxy[index]]

                                # New Center for bird view change
                                new_center = (int((x2 + x1) / 2), int((y2 + y1) / 2))

                                if not (
                                    downoid[0] > self.width
                                    or downoid[0] < 0
                                    or downoid[1] > self.height + 200
                                    or downoid[1] < 0
                                ):
                                    self.draw_boxes(
                                        frame,
                                        bbox_xyxy[index],
                                        self.classNames,
                                        object_id,
                                        identities,
                                        index=index,
                                        threshold=self.threshold,
                                        downoid=downoid,
                                        bird_view_image=self.bird_view_img,
                                    )

                                    # current_object_point = downoid[0], downoid[1]
                                    current_object_point = new_center[0], new_center[1]
                                    if current_object_point not in obj_points:
                                        obj_points.append(current_object_point)
                                        current_obj["object_point"] = (
                                            current_object_point
                                        )
                                        current_obj["object_id"] = (
                                            int(identities[index])
                                            if identities is not None
                                            else 0
                                        )
                                        try:
                                            current_obj["class"] = class_names[index]
                                            current_obj["state"] = (
                                                "unsafe" if self.violation else "safe"
                                            )
                                        except Exception as e:
                                            continue
                                        data_json["objects"].append(
                                            deepcopy(current_obj)
                                        )
                        elif self.use_case in ["waiting_area"]:
                            self.COUNT = 0
                            # Iterate over the transformed points
                            for index, downoid in enumerate(rotated_object_points_list):

                                # New Changes
                                x1, y1, x2, y2 = [int(i) for i in bbox_xyxy[index]]

                                # New Center for bird view change
                                new_center = (int((x2 + x1) / 2), int((y2 + y1) / 2))

                                top_left = (x1, y1)
                                bottom_right = (x2, y2)

                                top_right, bottom_left = generate_rectangle_coordinates(
                                    top_left, bottom_right
                                )

                                if (
                                    is_inside_bbox(bottom_left, self.zone_points)
                                    or is_inside_bbox(top_left, self.zone_points)
                                    or is_inside_bbox(top_right, self.zone_points)
                                    or is_inside_bbox(bottom_right, self.zone_points)
                                ):

                                    x, y = downoid
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

                                    try:
                                        if list_boxes[index][0]:
                                            self.COUNT += 1
                                    except Exception as e:
                                        pass
                                    # self.violation = True
                                    if not self.start_time:
                                        self.start_time = time.time()

                                    # current_object_point = downoid[0], downoid[1]
                                    current_object_point = new_center[0], new_center[1]
                                    if current_object_point not in obj_points:
                                        obj_points.append(current_object_point)
                                        current_obj["object_point"] = (
                                            current_object_point
                                        )
                                        current_obj["object_id"] = (
                                            int(identities[index])
                                            if identities is not None
                                            else 0
                                        )
                                        try:
                                            current_obj["class"] = class_names[index]
                                            current_obj["state"] = "safe"
                                        except Exception as e:
                                            continue
                                        data_json["objects"].append(
                                            deepcopy(current_obj)
                                        )

                except Exception as e:
                    logger.debug(f"Error = {str(e)}")

                if (
                    self.use_case in ["zone_intrusion", "water_edge", "waiting_area"]
                    or "traffic_rules" in self.use_case
                ):
                    camera_points = reverse_perspective_transform(
                        self.corner_points, width_og, height_og, frame, self.zone_points
                    )

                    cv2.polylines(
                        frame,
                        [self.zone_points],
                        isClosed=True,
                        color=(0, 0, 0),
                        thickness=2,
                    )

                    cv2.polylines(
                        self.bird_view_img,
                        [self.zone_points],
                        isClosed=True,
                        color=(0, 0, 0),
                        thickness=2,
                    )

                # cpu_usage = psutil.cpu_percent(interval=1)
                # print(f"CPU  usage is : {cpu_usage}")

                # pynvml.nvmlInit()
                # device_count = pynvml.nvmlDeviceGetCount()
                # if device_count == 0:
                #    print("GPU not found!")

                # if os.environ.get("DEBUG") == "1":
                #    cv2.imshow("Bird view", self.bird_view_img)
                #    cv2.imshow("Video", frame)
                # if cv2.waitKey(1) & 0xFF == ord("q"):
                #    break

                logger.debug(f"Calling Rebbit MQ")
                # logger.debug(f"Video list = {self.output_video_1}")
                # Write the both outputs video to a local folders
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
                    video_saving_queue=self.video_saving_queue
                )
                # time.sleep(0.5)

            else:
                logger.debug("Waiting for frames.... ")
                time.sleep(0.01)


def remove_lock_file(lock_file_path):
    if os.path.exists(lock_file_path):
        os.remove(lock_file_path)
        logger.debug(f"Lock file {lock_file_path} removed.")


def cleanup(capture_process, detect_process, stop_event, lock_file_path, video_saving_queue, video_saving_process):
    """Clean up resources on exit."""
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


def handle_supervisor_signals(signum, frame, stop_event):
    logger.debug(f"Received signal {signum}. Stopping processes...")
    stop_event.set()


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
    type_of_violation = use_case
    #frame_queue = mp.Queue(maxsize=10000)
    frame_queue = mp.Manager().Queue(maxsize=10000)
    stop_event = mp.Event()

    video_saving_queue = mp.Manager().Queue(maxsize=1000)

    detection = cls_Zone(type_of_violation, frame_queue, stop_event, video_saving_queue)

    # Define processes for frame capture and detection
    capture_process = mp.Process(target=detection.capture_frames)
    detect_process = mp.Process(target=detection.detect)
    video_saving_process = mp.Process(target=generate_video_and_save, args=(video_saving_queue, stop_event))

    # Register signal handlers
    def signal_handler(signum, frame):
        logger.debug(f"Received signal {signum}. Initiating cleanup.")
        cleanup(capture_process, detect_process, stop_event, lock_file_path, video_saving_queue, video_saving_process)
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
            #time.sleep(0.1)
    except Exception as e:
        logger.debug(f"Error occurred in Main: {str(e)}")
    finally:
        cleanup(capture_process, detect_process, stop_event, lock_file_path, video_saving_queue, video_saving_process)


if __name__ == "__main__":
    pass
    # main()
    # type_of_violation = "traffic_rules_no_entry"
    #
    # Detection = cls_Zone(type_of_violation)
    # Detection.start()

    # python thread_test.py 192.168.141.226
