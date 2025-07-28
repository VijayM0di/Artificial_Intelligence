type_of_violation = 'waiting_area'

from helpers.zone import cls_Zone, main

if __name__ == "__main__":
    main(type_of_violation)

#Detection = cls_Zone(type_of_violation)
#Detection.detect()
#
# import os
# import sys
# import numpy as np
# import imutils
# import time
# import math
# import cv2
# from copy import deepcopy
# from helpers.log_manager import error_logger, logger
# from helpers.preset_manager import set_preset, get_camera_calibration
# from helpers.zone_manager import fetch_zone_camera_use_case
# from settings import access_token, RTSP_URL_PATTERN, USE_CASES_RETRIEVAL_URL, USECASE_ABBR, WIDTH_OG, SIZE_FRAME, \
#     CALIBRATION_IMG_PATH, HEIGHT_OG
# import requests
# from settings import CAMERA_RETRIEVAL_URL
# import torch
# from ultralytics import YOLO
# from helpers.bird_view_transfo_functions import compute_perspective_transform, compute_point_perspective_transformation, \
#     BIG_CIRCLE, SMALL_CIRCLE, xyxy_to_xywh, get_centroids_and_groundpoints, reverse_perspective_transform, \
#     COLOR_GREEN
# from helpers.colors import bcolors
# from helpers.deepsort_manager import init_tracker
# from helpers.violations_manager import write_video_upload_minio, clear_existing_frames
#
# COUNTER = 0
# COUNT = 0
#
# type_of_violation = 'waiting_area'
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
# all_data = []
# logger.debug(f"Stream started on camera {camera_data['id']} for "
#              f"use case {usecase_data['id']}")
#
# # Loop until the end of the video stream
# while True:
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
#     (frame_exists, frame) = vs.read()
#     # Test if it has reached the end of the video
#     if not frame_exists:
#         error_logger.error("Camera frame missing!! Reloading the stream!!")
#         vs = cv2.VideoCapture(RTSP_URL)
#         continue
#     else:
#         # Resize the image to the correct size
#         frame = imutils.resize(frame, width=int(size_frame))
#
#         # describe the type of font to be used
#         font = cv2.FONT_HERSHEY_SIMPLEX
#         text_1 = 'Number of vehicles : ' +str(COUNT)
#         # insert text in teh video
#         cv2.putText(frame,
#                     text_1,
#                     (50,50),
#                     font,1,
#                     (0,255,255),
#                     2,
#                     cv2.LINE_4)
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
#                 if cls in [0, 1, 2, 3, 4, 5, 6, 7]:
#                     if conf > 0.5:
#                         b = [x1, y1, x2, y2]
#                         list_boxes.append((x1, y1, w, h))
#                         x_c, y_c, bbox_w, bbox_h = xyxy_to_xywh(*b)
#                         xywh_obj = [x_c, y_c, bbox_w, bbox_h]
#                         xywh_bboxs.append(xywh_obj)
#                         confidences.append(conf)
#                         oids.append(cls)
#                         pboxes.append((x1, y1, w, h, cls))
#                         class_names.append(classNames[cls])
#                         COUNTER = len(list_boxes)
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
#             COUNT = 0
#             # Iterate over the transformed points
#             for index, downoid in enumerate(rotated_object_points_list):
#                 if cv2.pointPolygonTest(zone_points, tuple(downoid), False) >= 0:
#                     # Point is inside the zone
#                     if not (downoid[0] > width or downoid[0] < 0 or downoid[1] > height + 200 or downoid[1] < 0):
#                         try:
#                             cv2.rectangle(frame, (bbox_xyxy[index][0], bbox_xyxy[index][1]),
#                                       (bbox_xyxy[index][2], bbox_xyxy[index][3]), COLOR_GREEN, 2)
#                         except Exception as e:
#                             continue
#                         x, y = downoid
#                         cv2.circle(bird_view_img, (int(x), int(y)), BIG_CIRCLE, COLOR_GREEN, 2)
#                         cv2.circle(bird_view_img, (int(x), int(y)), SMALL_CIRCLE, COLOR_GREEN, -1)
#                         try:
#                             if list_boxes[index][0]:
#                                 COUNT += 1
#                         except Exception as e:
#                             pass
#                         violation = True
#                         if not start_time:
#                             start_time = time.time()
#                         current_object_point = downoid[0], downoid[1]
#                         if current_object_point not in obj_points:
#                             obj_points.append(current_object_point)
#                             current_obj['object_point'] = current_object_point
#                             current_obj['object_id'] = int(identities[index]) if identities is not None else 0
#                             try:
#                                 current_obj['class'] = class_names[index]
#                                 current_obj['state'] = "safe"
#                             except Exception as e:
#                                 continue
#                             data_json['objects'].append(deepcopy(current_obj))
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