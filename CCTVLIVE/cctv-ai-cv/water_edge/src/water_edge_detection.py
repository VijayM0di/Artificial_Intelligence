type_of_violation = 'water_edge'

from helpers.zone import cls_Zone, main

if __name__ == "__main__":
    main(type_of_violation)

#Detection = cls_Zone(type_of_violation)
#Detection.detect()
#
# import os
# import sys
#
# from collections import deque
# from copy import deepcopy
# from ultralytics import YOLO
# from helpers.log_manager import logger, error_logger
# from helpers.preset_manager import set_preset, get_camera_calibration
# from helpers.zone_manager import fetch_zone_camera_use_case
# from read_from_rtsp import get_frame, frame_dir
# from settings import access_token, RTSP_URL_PATTERN, USE_CASES_RETRIEVAL_URL, USECASE_ABBR, WIDTH_OG, \
#     CALIBRATION_IMG_PATH, HEIGHT_OG, SIZE_FRAME
# import requests
# from settings import CAMERA_RETRIEVAL_URL
# import torch
# from helpers.bird_view_transfo_functions import compute_perspective_transform, compute_point_perspective_transformation, \
#     BIG_CIRCLE, SMALL_CIRCLE, COLOR_RED, COLOR_GREEN, xyxy_to_xywh, get_centroids_and_groundpoints, \
#     reverse_perspective_transform
# from helpers.colors import bcolors
# import numpy as np
# import imutils
# import time
# import math
# import cv2
# from helpers.deepsort_manager import init_tracker
# from helpers.predict import UI_box, data_deque, speed_line_queue, estimatespeed, compute_color_for_labels
# from helpers.violations_manager import write_video_upload_minio, clear_existing_frames
#
# if len(sys.argv) < 2:
#     error_logger.error("Camera IP address not provided")
#     sys.exit(-1)
# camera_ip = sys.argv[1]
#
# camera_req = requests.post(CAMERA_RETRIEVAL_URL,
#                            json={"advancedSearch": {"fields": ["ipAddress"],
#                                                     "keyword": camera_ip}},
#                            headers={'Authorization': f'Bearer {access_token}'})
# if camera_req.status_code != 200:
#     error_logger.error("There was some ERROR with the server!")
#     sys.exit(-1)
# cameras = camera_req.json()
# camera_data = None
# if cameras['data']:
#     camera_data = cameras['data'][0]
# else:
#     error_logger.error(f"Camera with IP {camera_ip} not found!!")
#     sys.exit(-1)
#
# logger.debug(f"Camera {camera_ip} Loaded!!")
#
# RTSP_URL = RTSP_URL_PATTERN.format(username=camera_data['username'],
#                                    password=camera_data['password'], ipAddress=camera_ip)
# use_case_req = requests.post(USE_CASES_RETRIEVAL_URL,
#                              json={"advancedSearch": {"fields": ["caseName"],
#                                                       "keyword": USECASE_ABBR[type_of_violation]}},
#                              headers={'Authorization': f'Bearer {access_token}'})
# if use_case_req.status_code != 200:
#     error_logger.error("There was some ERROR with the server!")
#     sys.exit(-1)
# usecases = use_case_req.json()
# usecase_data = usecases['data'][0]
# logger.debug(f"Use Case {usecase_data['id']} data retrieved!!")
#
#
# #########################################
# # Load the config for the top-down view #
# #########################################
# logger.debug(bcolors.WARNING + "[ Loading config for the bird view transformation ] " + bcolors.ENDC)
#
# width_og = WIDTH_OG
# height_og = HEIGHT_OG
# img_path = CALIBRATION_IMG_PATH.format(ip=camera_ip.replace('.', '_'))
# size_frame = SIZE_FRAME
# corner_points = get_camera_calibration(camera_data['id'])
# if not corner_points:
#     error_logger.error(f"Camera calibration points not found for camera {camera_ip}")
#     sys.exit(-1)
#
# logger.debug(bcolors.OKGREEN + " Done : [ Config loaded ] ..." + bcolors.ENDC)
#
# #########################################
# #     Compute transformation matrix		#
# #########################################
# # Compute  transformation matrix from the original frame
# image = cv2.imread(img_path)
# if image is None:
#     image = np.zeros((WIDTH_OG, HEIGHT_OG, 3), dtype=np.uint8)
#
# matrix, imgOutput = compute_perspective_transform(corner_points, width_og, height_og, image)
# height, width, _ = imgOutput.shape
# blank_image = np.zeros((height, width, 3), np.uint8)
# height = blank_image.shape[0]
# width = blank_image.shape[1]
# dim = (width, height)
# threshold = None
#
#
# # Load the bird's eye view image
# img = cv2.imread("img/chemin_1.png")
# bird_view_img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
#
# deepsort = init_tracker()
# logger.debug("DeepSort tracker initiated!!")
#
# set_preset(camera_ip, camera_data)
# logger.debug("Default Preset activated!!")
#
# start_time = None
#
# center = (dim[0] // 2, dim[1] // 2)
# rotation_matrix = cv2.getRotationMatrix2D(center, camera_data['orientationAngle'], 1.0)
#
# ######################################################
# #########									 #########
# # 				START THE VIDEO STREAM               #
# #########									 #########
# ######################################################
# # vs = cv2.VideoCapture('video/output_cam8.mp4')
# output_video_1, output_video_2 = None, None
#
# model_yolo = YOLO("models/yolov8x.pt")
# logger.debug("YOLO model loaded!!")
#
#
# zone_points = fetch_zone_camera_use_case(camera_data['id'], usecase_data['id'])
# if not zone_points:
#     error_logger.error(f"Zone not found for the use case: {usecase_data['id']} with camera: {camera_data['id']}")
#     sys.exit(-1)
# logger.debug("Zone Points retrieved!!")
#
# data_json = {'camera_id': camera_data['id'], 'camera_ip': camera_ip, 'zone_points': zone_points}
#
# # Convert the selected points to a NumPy array
# zone_points = np.array(zone_points, dtype=np.int32)
# vs = cv2.VideoCapture(RTSP_URL)
# # dict maping class_id to class_name
# classNames = model_yolo.model.names
#
# clear_existing_frames(type_of_violation)
# logger.debug("Existing frame data cleared!!")
#
#
# def draw_boxes(img, bbox, names, object_id, identities=None, index=None, offset=(0, 0), list_boxes=None,
#                confidences=None, classNames=None, oids=None, threshold=None, downoid=None, bird_view_image=None,
#                color=None):
#     global violation
#     global start_time
#     height, width, _ = img.shape
#     # remove tracked point from buffer if object is lost
#     for key in list(data_deque):
#         if key not in identities:
#             data_deque.pop(key)
#
#     x1, y1, x2, y2 = [int(i) for i in bbox]
#     x1 += offset[0]
#     x2 += offset[0]
#     y1 += offset[1]
#     y2 += offset[1]
#
#     # code to find center of bottom edge
#     center = (int((x2 + x1) / 2), int((y2 + y2) / 2))
#
#     # get ID of object
#     id = int(identities[index]) if identities is not None else 0
#
#     # create new buffer for new object
#     if id not in data_deque:
#         data_deque[id] = deque(maxlen=64)
#         speed_line_queue[id] = []
#     color = color or compute_color_for_labels(object_id[index])
#     obj_name = names[object_id[index]]
#     label = '{}{:d}'.format("", id) + ":" + '%s' % (obj_name)
#
#     # add center to buffer
#     data_deque[id].appendleft(center)
#     if len(data_deque[id]) >= 2:
#         object_speed = estimatespeed(data_deque[id][1], data_deque[id][0])
#         speed_line_queue[id].append(object_speed)
#
#     # draw trail
#     for i in range(1, len(data_deque[id])):
#         # check if on buffer value is none
#         if data_deque[id][i - 1] is None or data_deque[id][i] is None:
#             continue
#         violation = True
#         if not start_time:
#             start_time = time.time()
#         x, y = downoid
#         cv2.circle(bird_view_image, (int(x), int(y)), BIG_CIRCLE, color, 2)
#         cv2.circle(bird_view_image, (int(x), int(y)), SMALL_CIRCLE, color, -1)
#         UI_box(bbox, img, label=label, color=color, line_thickness=2)
#
#     return img
#
#
# all_data = []
# logger.debug(f"Stream started on camera {camera_data['id']} for "
#              f"use case {usecase_data['id']}")
#
# # Loop until the end of the video stream
#
# frame_filename = get_frame(camera_ip)
# while frame_filename:
#     data_json['objects'] = []
#     current_obj = {}
#     obj_points = []
#
#     # Load the image of the ground and resize it to the correct size
#     img = cv2.imread("img/chemin_1.png")
#     bird_view_img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
#     confidences = []
#     pboxes = []
#     # Load the frame
#     frame = cv2.imread(f'{frame_dir}/{frame_filename}')
#     if frame is None:
#         frame_filename = get_frame(camera_ip)
#         continue
#     else:
#         # Resize the image to the correct size
#         frame = imutils.resize(frame, width=int(size_frame))
#
#         results = model_yolo(frame)
#         violation = False
#         list_boxes = []
#         xywh_bboxs = []
#         class_names = []
#         confs = []
#         oids = []
#         for r in results:
#             boxes = r.boxes
#             for box in boxes:
#                 # Bounding Box
#                 x1, y1, x2, y2 = box.xyxy[0]
#
#                 x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
#                 w, h = x2 - x1, y2 - y1
#
#                 # Confidence
#                 conf = math.ceil((box.conf[0] * 100)) / 100
#                 # Class Name
#                 cls = int(box.cls[0])
#
#                 currentClass = classNames[cls]
#                 if conf > 0.5:
#                     b = [x1, y1, x2, y2]
#                     list_boxes.append((x1, y1, w, h))
#                     x_c, y_c, bbox_w, bbox_h = xyxy_to_xywh(*b)
#                     xywh_obj = [x_c, y_c, bbox_w, bbox_h]
#                     xywh_bboxs.append(xywh_obj)
#                     confidences.append(conf)
#                     oids.append(cls)
#                     class_names.append(classNames[cls])
#                     pboxes.append((x1, y1, w, h, cls))
#
#         xywhs = torch.Tensor(xywh_bboxs)
#         confss = torch.Tensor(confidences)
#
#         try:
#             outputs = deepsort.update(xywhs, confss, oids, frame)
#         except Exception as e:
#             continue
#
#         if len(outputs) > 0:
#             bbox_xyxy = outputs[:, :4]
#             identities = outputs[:, -2]
#             object_id = outputs[:, -1]
#
#             array_centroids, array_groundpoints = get_centroids_and_groundpoints(bbox_xyxy)
#
#             # Use the transform matrix to get the transformed coordonates
#             transformed_downoids = compute_point_perspective_transformation(matrix, array_groundpoints)
#             rotated_object_points = cv2.transform(np.array(transformed_downoids).reshape(-1, 1, 2),
#                                                   rotation_matrix)
#             if rotated_object_points is None:
#                 continue
#             rotated_object_points_list = [(point[0][0], point[0][1]) for point in rotated_object_points]
#
#             # Iterate over the transformed points
#             for index, downoid in enumerate(rotated_object_points_list):
#                 if cv2.pointPolygonTest(zone_points, tuple(downoid), False) >= 0:
#                     # Point is inside the zone
#                     if not (downoid[0] > width or downoid[0] < 0 or downoid[1] > height + 200 or downoid[1] < 0):
#                         x, y = downoid
#                         draw_boxes(frame, bbox_xyxy[index], classNames, object_id, identities, index=index,
#                                    list_boxes=list_boxes,
#                                    confidences=confidences, threshold=threshold, classNames=classNames, oids=oids,
#                                    downoid=downoid, bird_view_image=bird_view_img, color=COLOR_RED)
#                         current_object_point = downoid[0], downoid[1]
#                         if current_object_point not in obj_points:
#                             obj_points.append(current_object_point)
#                             current_obj['object_point'] = current_object_point
#                             current_obj['class'] = classNames[object_id[index]]
#                             current_obj['state'] = 'unsafe'
#                             current_obj['object_id'] = int(identities[index]) if identities is not None else 0
#                             data_json['objects'].append(deepcopy(current_obj))
#                 else:
#                     draw_boxes(frame, bbox_xyxy[index], classNames, object_id, identities, index=index,
#                                list_boxes=list_boxes,
#                                confidences=confidences, threshold=threshold, classNames=classNames, oids=oids,
#                                downoid=downoid, bird_view_image=bird_view_img, color=COLOR_GREEN)
#                     current_object_point = downoid[0], downoid[1]
#                     if current_object_point not in obj_points:
#                         obj_points.append(current_object_point)
#                         current_obj['object_point'] = current_object_point
#                         current_obj['class'] = classNames[object_id[index]]
#                         current_obj['state'] = 'safe'
#                         current_obj['object_id'] = int(identities[index]) if identities is not None else 0
#                         data_json['objects'].append(deepcopy(current_obj))
#
#         camera_points = reverse_perspective_transform(corner_points, width_og, height_og, frame, zone_points)
#         # Draw the cube representation on the camera view
#         for i in range(4):
#             pt1 = tuple(map(int, camera_points[i][0]))
#             pt2 = tuple(map(int, camera_points[(i + 1) % 4][0]))
#             pt3 = tuple(map(int, camera_points[i - 4][0]))
#             cv2.line(frame, pt1, pt2, (0, 0, 0), 2)
#             cv2.line(frame, pt1, pt3, (0, 0, 0), 2)
#             cv2.line(frame, pt2, pt3, (0, 0, 0), 2)
#
#         cv2.polylines(bird_view_img, [zone_points], isClosed=True, color=(0, 0, 0), thickness=2)
#
#         if os.environ.get('DEBUG') == '1':
#             cv2.imshow("Bird view", bird_view_img)
#             cv2.imshow("Video", frame)
#         if cv2.waitKey(1) & 0xFF == ord("q"):
#             break
#
#     # Write the both outputs video to a local folders
#     output_video_1, start_time, all_data = write_video_upload_minio(frame, data_json, output_video_1, violation,
#                                                                     type_of_violation, start_time,  usecase_data['id'],
#                                                                     all_data)