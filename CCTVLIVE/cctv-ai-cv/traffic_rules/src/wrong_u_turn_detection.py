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
    UC_WRONG_TURN_MIN_RATIO,
)

import torch

from helpers.colors import bcolors
import numpy as np
import imutils
import time
import math
import cv2
from helpers.deepsort_manager import init_tracker
from helpers.violations_manager import (
    write_video_upload_minio_wrong_turn,
    generate_video_and_save,
)

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


def calculate_angle(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    angle = math.atan2(y2 - y1, x2 - x1)
    return math.degrees(angle)


def determine_entry_side(center_point, zone_points) -> str:
    x, y = center_point
    bottom_left, top_left, top_right, bottom_right = zone_points
    # top_right, bottom_right, bottom_left, top_left = zone_points

    x1, y1 = top_left
    x2, y2 = top_right
    x3, y3 = bottom_left
    x4, y4 = bottom_right

    # Left
    if (x < x1 or x < x3) and y1 < y < y2:
        return "left"
    # Bottom
    elif x3 < x < x4 and (y > y3 or y > y4):
        return "bottom"
    # Right
    elif (x > x2 or x > x4) and y2 < y < y4:
        return "right"
    # Top
    elif x1 < x < x2 and (y < y1 or y < y2):
        return "top"
    else:
        return "inside"


def determine_position_if_already_inside(initial_coordinates, new_coordinates) -> str:
    x, y = initial_coordinates

    (
        x1,
        y1,
    ) = new_coordinates

    angle = math.atan2(y1 - y, x1 - x)
    degree = math.degrees(angle)

    if 45 < degree < 135:
        return "top"
    elif -45 < degree < 45:
        return "left"
    elif -135 < degree < -45:
        return "bottom"
    elif -180 <= degree <= -135 or 135 <= degree <= 180:
        return "right"
    else:
        return "inside"


def get_top_min_percent_line(zone_points):
    bottom_left, top_left, top_right, bottom_right = zone_points
    # top_right, bottom_right, bottom_left, top_left = zone_points

    x1, y1 = top_left
    x2, y2 = bottom_left

    left_x = min([x1, x2])
    left_y = y1 + ((y2 - y1) * UC_WRONG_TURN_MIN_RATIO)

    x3, y3 = top_right
    x4, y4 = bottom_right

    right_x = max([x3, x4])
    right_y = y3 + ((y4 - y3) * UC_WRONG_TURN_MIN_RATIO)

    return (int(left_x), int(left_y)), (int(right_x), int(right_y))


def get_left_min_percent_line(zone_points):
    bottom_left, top_left, top_right, bottom_right = zone_points
    # top_right, bottom_right, bottom_left, top_left = zone_points

    x1, y1 = top_left
    x2, y2 = top_right

    left_top_x = x1 + ((x2 - x1) * UC_WRONG_TURN_MIN_RATIO)
    left_top_y = min([y1, y2])

    x3, y3 = bottom_left
    x4, y4 = bottom_right

    left_bottom_x = x3 + ((x4 - x3) * UC_WRONG_TURN_MIN_RATIO)
    left_bottom_y = max([y3, y4])

    return (int(left_top_x), int(left_top_y)), (int(left_bottom_x), int(left_bottom_y))


def get_bottom_min_percent_line(zone_points):
    bottom_left, top_left, top_right, bottom_right = zone_points
    # top_right, bottom_right, bottom_left, top_left = zone_points

    x1, y1 = bottom_left
    x2, y2 = top_left

    bottom_left_x = min([x1, x2])
    bottom_left_y = y1 + ((y2 - y1) * UC_WRONG_TURN_MIN_RATIO)

    x3, y3 = bottom_right
    x4, y4 = top_right

    bottom_right_x = max([x3, x4])
    bottom_right_y = y3 + ((y4 - y3) * UC_WRONG_TURN_MIN_RATIO)

    return (int(bottom_left_x), int(bottom_left_y)), (
        int(bottom_right_x),
        int(bottom_right_y),
    )


def get_right_min_percent_line(zone_points):
    bottom_left, top_left, top_right, bottom_right = zone_points
    # top_right, bottom_right, bottom_left, top_left = zone_points

    x1, y1 = top_right
    x2, y2 = top_left

    right_top_x = x1 + ((x2 - x1) * UC_WRONG_TURN_MIN_RATIO)
    right_top_y = min([y1, y2])

    x3, y3 = bottom_right
    x4, y4 = bottom_left

    right_bottom_x = x3 + ((x4 - x3) * UC_WRONG_TURN_MIN_RATIO)
    right_bottom_y = max([y3, y4])

    return (int(right_top_x), int(right_top_y)), (
        int(right_bottom_x),
        int(right_bottom_y),
    )


def line_intersects_rect(line_start, line_end, rect_points):
    def line_intersection(line1_start, line1_end, line2_start, line2_end):
        # Convert points to numpy arrays
        p = np.array(line1_start)
        r = np.array(line1_end) - p
        q = np.array(line2_start)
        s = np.array(line2_end) - q

        r_cross_s = np.cross(r, s)
        q_minus_p_cross_r = np.cross((q - p), r)

        if r_cross_s == 0:
            return None  # Lines are parallel or collinear

        t = np.cross((q - p), s) / r_cross_s
        u = q_minus_p_cross_r / r_cross_s

        if 0 <= t <= 1 and 0 <= u <= 1:
            intersection = p + t * r
            return intersection.tolist()
        else:
            return None  # No intersection

    rect_edges = [
        (rect_points[0], rect_points[1]),
        (rect_points[1], rect_points[2]),
        (rect_points[2], rect_points[3]),
        (rect_points[3], rect_points[0]),
    ]

    intersections = []
    for edge in rect_edges:
        intersect_point = line_intersection(line_start, line_end, edge[0], edge[1])
        if intersect_point:
            intersections.append((int(intersect_point[0]), int(intersect_point[1])))

    if len(intersections) == 2:
        # Determine top and bottom points based on y-coordinate
        intersections.sort(key=lambda pt: pt[1])
        return intersections[0], intersections[1]
    else:
        return None  # Line does not intersect rectangle at exactly two points


def find_intersection(line1, line2):
    def line_equation(p1, p2):
        """Return A, B, C of line equation Ax + By = C from points p1 and p2."""
        A = p2[1] - p1[1]
        B = p1[0] - p2[0]
        C = A * p1[0] + B * p1[1]
        return A, B, C

    A1, B1, C1 = line_equation(*line1)
    A2, B2, C2 = line_equation(*line2)

    determinant = A1 * B2 - A2 * B1

    if determinant == 0:
        return None  # Lines are parallel or collinear

    x = (B2 * C1 - B1 * C2) / determinant
    y = (A1 * C2 - A2 * C1) / determinant

    # Check if the intersection point is within the line segments
    def is_between(p, q, r):
        """Check if point q is between points p and r."""
        return min(p[0], r[0]) <= q[0] <= max(p[0], r[0]) and min(p[1], r[1]) <= q[
            1
        ] <= max(p[1], r[1])

    if is_between(line1[0], (x, y), line1[1]) and is_between(
        line2[0], (x, y), line2[1]
    ):
        return int(x), int(y)
    else:
        return None  # Intersection point is not within the line segments


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
        self.waiting_obj = {}
        self.camera_ip = ""
        self.output_video_1 = None
        self.output_video_frame_list = []
        self.zone_points = []
        self.raw_zone_points = []

        # New Logic Functions
        self.turning_avg = {}
        self.initial_center_points = {}
        self.centroid_values = {}
        self.output_video_frames = {}
        self.violation_video_frames = {}

        self.top_rect_points = None
        self.right_rect_points = None
        self.left_rect_points = None
        self.bottom_rect_points = None
        self.internal_zone_points = None


        yt_video_url = "https://www.youtube.com/live/6dp-bvQ7RWo?si=bE8v3svt3sKLEtfQ"
        #self.stream_url = get_youtube_live_stream_url(yt_video_url)
        self.stream_url = '/app/videos/Part-2/wrong-turn/U-turn-file2.mp4'
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
        list_boxes=None,
        confidences=None,
        classNames=None,
        oids=None,
        threshold=None,
        downoid=None,
        bird_view_image=None,
    ):
        try:
            height, width, _ = img.shape
            # remove tracked point from buffer if object is lost
            #for key in list(self.data_deque):
            #    if key not in identities:
            #        self.data_deque.pop(key)
            #        self.output_video_frames.pop(key)
            #        self.violation_video_frames.pop(key)

            x1, y1, x2, y2 = [int(i) for i in bbox]

            target_center = (int((x2 + x1) / 2), int((y2 + y1) / 2))

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
                self.output_video_frames[id] = []
                self.violation_video_frames[id] = []

            color = compute_color_for_labels(object_id[index])
            obj_name = names[object_id[index]]
            if obj_name != "person":
                obj_name = "vehicle"
            label = "{}{:d}".format("", id) + ":" + "%s" % (obj_name)

            # add center to buffer
            self.centroid_values[id].append(target_center)
            self.data_deque[id].appendleft(target_center)
            if len(self.data_deque[id]) >= 2:
                object_speed = estimatespeed(
                    self.data_deque[id][1], self.data_deque[id][0]
                )
                self.speed_line_queue[id].append(object_speed)

            try:
                avg_speed = sum(self.speed_line_queue[id]) // len(
                    self.speed_line_queue[id]
                )
                if avg_speed <= 3:
                    return img
            except:
                pass

            # draw trail
            for i in range(1, len(self.data_deque[id])):
                # check if on buffer value is none
                if self.data_deque[id][i - 1] is None or self.data_deque[id][i] is None:
                    continue

                # Analyze the tracked trajectories for wrong U-turn patterns
                for obj_id, centroids in self.data_deque.items():
                    if len(centroids) > 1 and obj_id in identities:

                        initial_center = self.centroid_values[id][0]

                        if len(self.centroid_values[id]) > 50:
                            self.centroid_values[id] = self.centroid_values[id][-50:]

                        angle = calculate_angle(initial_center, target_center)
                        # print(f"{id} Angle = {angle}")

                        entry_side = self.turning_avg[id]["entry"]
                        label += " : " + entry_side
                        # logger.debug(f"{id} Entry = {entry_side}")

                        # logger.debug(
                        #    f"{id} Entered Origin = {self.turning_avg[id]['entered_origin']}"
                        # )
                        # logger.debug(
                        #    f"{id} Passed Origin = {self.turning_avg[id]['passed_origin']}"
                        # )
                        # logger.debug(f"*************************")

                        if self.turning_avg[id]["partial_violation"]:
                            self.turning_avg[id]["frames"].append(target_center)

                        if entry_side == "inside":
                            centroid = centroids[0]
                            new_entry_side = determine_position_if_already_inside(
                                self.initial_center_points[identities[index]], centroid
                            )
                            self.turning_avg[id]["entry"] = new_entry_side
                            entry_side = new_entry_side

                        if entry_side == "right":
                            # New
                            if 90 < angle < 158 or -90 > angle > -158:
                                if not self.turning_avg[id]["has_turned"]:
                                    self.turning_avg[id]["has_turned"] = True
                                    self.turning_avg[id]["partial_violation"] = True
                            else:
                                if (
                                    self.turning_avg[id]["has_turned"]
                                    and len(self.turning_avg[id]["frames"]) <= 20
                                ):
                                    self.turning_avg[id]["has_turned"] = False
                                    self.turning_avg[id]["partial_violation"] = False
                            # //

                            if (
                                is_inside_bbox(target_center, self.right_rect_points)
                                and self.turning_avg[id]["entered_origin"] == False
                            ):
                                if not self.turning_avg[id]["entered_origin"]:
                                    self.turning_avg[id]["entered_origin"] = True
                            if (
                                is_inside_bbox(
                                    target_center, self.internal_zone_points
                                )
                                and self.turning_avg[id]["entered_origin"] == True
                            ):
                                if not self.turning_avg[id]["passed_origin"]:
                                    self.turning_avg[id]["passed_origin"] = True
                            # if is_inside_bbox(target_center, top_rect_points) or is_inside_bbox(target_center, bottom_rect_points) or (
                            if (
                                is_inside_bbox(target_center, self.right_rect_points)
                                and self.turning_avg[id]["passed_origin"] == True
                                and self.turning_avg[id]["has_turned"] == True
                            ):
                                if not self.turning_avg[id]["violation"]:
                                    self.turning_avg[id]["violation"] = True
                            else:
                                if self.turning_avg[id]["violation"]:
                                    self.turning_avg[id]["violation"] = False

                        elif entry_side == "left":
                            # New
                            if -90 < angle < -23 or 90 > angle > 23:
                                if not self.turning_avg[id]["has_turned"]:
                                    self.turning_avg[id]["has_turned"] = True
                                    self.turning_avg[id]["partial_violation"] = True
                            else:
                                if (
                                    self.turning_avg[id]["has_turned"]
                                    and len(self.turning_avg[id]["frames"]) <= 20
                                ):
                                    self.turning_avg[id]["has_turned"] = False
                                    self.turning_avg[id]["partial_violation"] = False
                            # ////

                            if (
                                is_inside_bbox(target_center, self.left_rect_points)
                                and self.turning_avg[id]["entered_origin"] == False
                            ):
                                if not self.turning_avg[id]["entered_origin"]:
                                    self.turning_avg[id]["entered_origin"] = True
                            if (
                                is_inside_bbox(target_center, self.internal_zone_points)
                                and self.turning_avg[id]["entered_origin"] == True
                            ):
                                if not self.turning_avg[id]["passed_origin"]:
                                    self.turning_avg[id]["passed_origin"] = True
                            # if is_inside_bbox(target_center, top_rect_points) or is_inside_bbox(target_center, bottom_rect_points) or (
                            if (
                                is_inside_bbox(target_center, self.left_rect_points)
                                and self.turning_avg[id]["passed_origin"] == True
                                and self.turning_avg[id]["has_turned"] == True
                            ):
                                if not self.turning_avg[id]["violation"]:
                                    self.turning_avg[id]["violation"] = True
                            else:
                                if self.turning_avg[id]["violation"]:
                                    self.turning_avg[id]["violation"] = False

                        elif entry_side == "bottom":

                            # New
                            if 112 < angle < 180 or 68 > angle > 0:
                                if not self.turning_avg[id]["has_turned"]:
                                    self.turning_avg[id]["has_turned"] = True
                                    self.turning_avg[id]["partial_violation"] = True
                            else:
                                if (
                                    self.turning_avg[id]["has_turned"]
                                    and len(self.turning_avg[id]["frames"]) <= 20
                                ):
                                    self.turning_avg[id]["has_turned"] = False
                                    self.turning_avg[id]["partial_violation"] = False
                            # ///

                            if (
                                is_inside_bbox(target_center, self.bottom_rect_points)
                                and self.turning_avg[id]["entered_origin"] == False
                            ):
                                if not self.turning_avg[id]["entered_origin"]:
                                    self.turning_avg[id]["entered_origin"] = True
                            if (
                                is_inside_bbox(
                                    target_center, self.internal_zone_points
                                )
                                and self.turning_avg[id]["entered_origin"] == True
                            ):
                                if not self.turning_avg[id]["passed_origin"]:
                                    self.turning_avg[id]["passed_origin"] = True
                            # if is_inside_bbox(target_center, left_rect_points) or is_inside_bbox(target_center, right_rect_points) or (
                            if (
                                is_inside_bbox(target_center, self.bottom_rect_points)
                                and self.turning_avg[id]["passed_origin"] == True
                                and self.turning_avg[id]["has_turned"] == True
                            ):
                                if not self.turning_avg[id]["violation"]:
                                    self.turning_avg[id]["violation"] = True
                            else:
                                if self.turning_avg[id]["violation"]:
                                    self.turning_avg[id]["violation"] = False

                        elif entry_side == "top":
                            # New
                            if -68 < angle < 0 or -112 > angle > -180:
                                if not self.turning_avg[id]["has_turned"]:
                                    self.turning_avg[id]["has_turned"] = True
                                    self.turning_avg[id]["partial_violation"] = True
                            else:
                                if (
                                    self.turning_avg[id]["has_turned"]
                                    and len(self.turning_avg[id]["frames"]) <= 20
                                ):
                                    self.turning_avg[id]["has_turned"] = False
                                    self.turning_avg[id]["partial_violation"] = False
                            # ////

                            if (
                                is_inside_bbox(target_center, self.top_rect_points)
                                and self.turning_avg[id]["entered_origin"] == False
                            ):
                                if not self.turning_avg[id]["entered_origin"]:
                                    self.turning_avg[id]["entered_origin"] = True
                            if (
                                is_inside_bbox(target_center, self.internal_zone_points)
                                and self.turning_avg[id]["entered_origin"] == True
                            ):
                                if not self.turning_avg[id]["passed_origin"]:
                                    self.turning_avg[id]["passed_origin"] = True
                            # if is_inside_bbox(target_center, left_rect_points) or is_inside_bbox(target_center, right_rect_points) or (
                            if (
                                is_inside_bbox(target_center, self.top_rect_points)
                                and self.turning_avg[id]["passed_origin"] == True
                                and self.turning_avg[id]["has_turned"] == True
                            ):
                                if not self.turning_avg[id]["violation"]:
                                    self.turning_avg[id]["violation"] = True
                            else:
                                if self.turning_avg[id]["violation"]:
                                    self.turning_avg[id]["violation"] = False

                        # Detect wrong U-turns based on the change in direction
                        if self.turning_avg[id]["violation"]:
                            self.violation = True
                            if not self.start_time:
                                self.start_time = time.time()

                            # x, y = downoid
                            x, y = target_center
                            cv2.circle(
                                bird_view_image,
                                (int(x), int(y)),
                                BIG_CIRCLE,
                                COLOR_RED,
                                2,
                            )
                            cv2.circle(
                                bird_view_image,
                                (int(x), int(y)),
                                SMALL_CIRCLE,
                                COLOR_RED,
                                -1,
                            )
                            #logger.debug("U-turn detected for Object ID:", id)
                            UI_box(
                                bbox,
                                img,
                                label=label,
                                color=COLOR_RED,
                                line_thickness=2,
                            )

                            # Draw the trail for objects with U-turns
                            for i in range(1, len(centroids)):
                                # Draw the trail line segment
                                cv2.line(
                                    img, centroids[i - 1], centroids[i], (0, 0, 255), 2
                                )

                            self.violation_video_frames[id].append(img)

                        else:

                            UI_box(
                                bbox,
                                img,
                                label=label,
                                color=COLOR_GREEN,
                                line_thickness=2,
                            )

                            # Draw the trail for objects with U-turns
                            for i in range(1, len(centroids)):
                                # Draw the trail line segment
                                cv2.line(
                                    img, centroids[i - 1], centroids[i], (0, 255, 0), 2
                                )

            self.output_video_frames[id].append(img)
            return img

        except Exception as e:
            logger.debug(f"Error in draw_boxes function: {e}")

    def capture_frames(self):

        # Debug Video frames
        cap = cv2.VideoCapture(self.stream_url)
        # cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        # cap.set(cv2.CAP_PROP_FPS, 25)
        # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 405)
        time.sleep(20)

        count = 0

        while not self.stop_event.is_set():

            # Debug video frames
            ret, frame = cap.read()
            if not ret:
                print("Camera frame missing!! Reloading the stream!!")
                cap = cv2.VideoCapture(self.stream_url)
                continue

            count += 1

            # RTSP stream
            # frame = process_frames(PUB_PORT, self.subscribed_topic)
            # if frame is None:
            #    logger.debug(f"frame is EMPTY!")
            #    continue

            if not self.frame_queue.full():
                if not count % 5 == 0:
                    self.frame_queue.put(frame)

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

        self.raw_zone_points = self.zone_points

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

        # New Calculations

        # Top Line

        new_top_left, new_top_right = get_top_min_percent_line(self.zone_points)

        top_result_right, top_result_left = line_intersects_rect(
            new_top_left, new_top_right, self.zone_points
        )

        # Left Line

        new_top_for_left, new_bottom_for_left = get_left_min_percent_line(
            self.zone_points
        )

        left_result_top, left_result_bottom = line_intersects_rect(
            new_top_for_left, new_bottom_for_left, self.zone_points
        )

        # Bottom Line

        new_bottom_left_for_bottom, new_bottom_right_for_bottom = (
            get_bottom_min_percent_line(self.zone_points)
        )

        bottom_result_right, bottom_result_left = line_intersects_rect(
            new_bottom_left_for_bottom, new_bottom_right_for_bottom, self.zone_points
        )

        # Right Line

        new_top_right_for_right, new_bottom_right_for_right = (
            get_right_min_percent_line(self.zone_points)
        )

        right_result_top, right_result_bottom = line_intersects_rect(
            new_top_right_for_right, new_bottom_right_for_right, self.zone_points
        )

        # Overlapping

        # For Right
        right_top_overlap = find_intersection(
            (top_result_left, top_result_right), (right_result_top, right_result_bottom)
        )

        right_bottom_overlap = find_intersection(
            (bottom_result_left, bottom_result_right),
            (right_result_top, right_result_bottom),
        )

        # For left

        left_top_overlap = find_intersection(
            (top_result_left, top_result_right), (left_result_top, left_result_bottom)
        )

        left_bottom_overlap = find_intersection(
            (bottom_result_left, bottom_result_right),
            (left_result_top, left_result_bottom),
        )

        # -----------------

        # The 4 Sub Zones for Road UC_WRONG_TURN_MIN_RATIO %

        self.top_rect_points = (
            left_top_overlap,
            left_result_top,
            right_result_top,
            right_top_overlap,
        )

        self.bottom_rect_points = (
            left_bottom_overlap,
            left_result_bottom,
            right_result_bottom,
            right_bottom_overlap,
        )

        #self.left_rect_points = (
        #    bottom_result_left,
        #    top_result_left,
        #    left_top_overlap,
        #    left_bottom_overlap,
        #)

        self.left_rect_points = (
            bottom_result_right,
            top_result_right,
            left_top_overlap,
            left_bottom_overlap,
        )


        self.right_rect_points = (
            right_bottom_overlap,
            right_top_overlap,
            top_result_right,
            bottom_result_right,
        )

        self.internal_zone_points = (
            right_top_overlap,
            left_top_overlap,
            left_bottom_overlap,
            right_bottom_overlap
        )

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
            "zone_points": self.raw_zone_points,
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

        while self.to_be_checked:
            logger.debug(f"Queue = {self.frame_queue.qsize()}")
            if not self.frame_queue.empty():
                frame = self.frame_queue.get()
                logger.debug("Got frame.... ")

                # Resize the image to the correct size
                frame = imutils.resize(frame, width=int(size_frame))

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
                        if cls not in VEHICLE_CLASS_ID:
                            continue
                        currentClass = self.classNames[cls]
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

                        # Iterate over the transformed points
                        for index, downoid in enumerate(rotated_object_points_list):
                            self.violation = False
                            color = COLOR_GREEN

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

                                if (
                                    int(identities[index])
                                    not in self.initial_center_points
                                ):
                                    self.initial_center_points[identities[index]] = (
                                        new_center
                                    )
                                    entry = determine_entry_side(
                                        self.initial_center_points[identities[index]],
                                        self.zone_points,
                                    )
                                    self.turning_avg[identities[index]] = {
                                        "violation": False,
                                        "partial_violation": False,
                                        "entry": entry,
                                        "on_border": False,
                                        "passed_origin": False,
                                        "has_turned": False,
                                        "entered_origin": False,
                                        "violation_frames": [],
                                        "frames": [],
                                    }
                                    self.centroid_values[identities[index]] = [
                                        new_center
                                    ]

                                x, y = downoid
                                self.draw_boxes(
                                    frame,
                                    bbox_xyxy[index],
                                    self.classNames,
                                    object_id,
                                    identities,
                                    index=index,
                                    list_boxes=list_boxes,
                                    confidences=confidences,
                                    threshold=self.threshold,
                                    classNames=self.classNames,
                                    oids=oids,
                                    downoid=downoid,
                                    bird_view_image=self.bird_view_img,
                                )

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

                except Exception as e:
                    logger.debug(f"Error in Detect loop {e}")

                #camera_points = reverse_perspective_transform(
                #    self.corner_points, width_og, height_og, frame, self.zone_points
                #)

                #cv2.polylines(
                #    frame,
                #    [np.array(self.top_rect_points, dtype=np.int32)],
                #    isClosed=True,
                #    color=(0, 128, 255),
                #    thickness=1,
                #)

                #cv2.polylines(
                #    frame,
                #    [np.array(self.bottom_rect_points, dtype=np.int32)],
                #    isClosed=True,
                #    color=(0, 128, 255),
                #    thickness=1,
                #)

                cv2.polylines(
                    frame,
                    [np.array(self.left_rect_points, dtype=np.int32)],
                    isClosed=True,
                    color=(0, 128, 255),
                    thickness=1,
                )

                #cv2.polylines(
                #    frame,
                #    [np.array(self.right_rect_points, dtype=np.int32)],
                #    isClosed=True,
                #    color=(0, 128, 255),
                #    thickness=1,
                #)

                #cv2.polylines(
                #    frame,
                #    [np.array(self.internal_zone_points, dtype=np.int32)],
                #    isClosed=True,
                #    color=(0, 0, 255),
                #    thickness=5,
                #)

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

                # # Show both images
                if os.environ.get("DEBUG") == "1":
                    cv2.imshow("Bird view", self.bird_view_img)
                    cv2.imshow("Video", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

                logger.debug(f"Calling Rebbit MQ")

                (
                    self.output_video_1,
                    self.start_time,
                    self.all_data,
                    self.output_video_frame_list,
                    self.output_video_frames,
                    self.violation_video_frames,
                ) = write_video_upload_minio_wrong_turn(
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
                    output_video_frames=self.output_video_frames,
                    violation_video_frames=self.violation_video_frames,
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
        # sys.exit(1)
    with open(lock_file_path, "w") as lock_file:
        lock_file.write(str(os.getpid()))

    # Ensure lock file is removed on exit
    atexit.register(remove_lock_file, lock_file_path)

    logger.debug("Calling main function")
    mp.set_start_method("spawn", force=True)

    frame_queue = mp.Queue(maxsize=1000)
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
    type_of_violation = "traffic_rules_wrong_turn"
    main(type_of_violation)
    #
    # Detection = cls_Zone(type_of_violation)
    # Detection.start()

    # python thread_test.py 192.168.141.226
