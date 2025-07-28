import numpy as np
import cv2

COLOR_RED = (0, 0, 255)
COLOR_GREEN = (0, 255, 0)
COLOR_BLUE = (255, 0, 0)
BIG_CIRCLE = 60
SMALL_CIRCLE = 3


def compute_perspective_transform(corner_points, width, height, image):
    """ Compute the transformation matrix
    @ corner_points : 4 corner points selected from the image
    @ height, width : size of the image
    """
    # Create an array out of the 4 corner points
    corner_points_array = np.float32(corner_points)
    # Create an array with the parameters (the dimensions) required to build the matrix
    img_params = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    # Compute and return the transformation matrix
    matrix = cv2.getPerspectiveTransform(corner_points_array, img_params)
    img_transformed = cv2.warpPerspective(image, matrix, (width, height))
    return matrix, img_transformed


def compute_point_perspective_transformation(matrix, list_downoids):
    """ Apply the perspective transformation to every ground point which have been detected on the main frame.
    @ matrix : the 3x3 matrix 
    @ list_downoids : list that contains the points to transform
    return : list containing all the new points
    """
    # Compute the new coordinates of our points
    list_points_to_detect = np.float32(list_downoids).reshape(-1, 1, 2)
    transformed_points = cv2.perspectiveTransform(list_points_to_detect, matrix)
    # Loop over the points and add them to the list that will be returned
    transformed_points_list = list()
    if transformed_points is not None:
        for i in range(0, transformed_points.shape[0]):
            transformed_points_list.append([transformed_points[i][0][0], transformed_points[i][0][1]])
    return transformed_points_list


def reverse_perspective_transform(corner_points, width, height, frame, zone_points):
    corner_points_array = np.float32(corner_points)
    # Create an array with the parameters (the dimensions) required to build the matrix
    img_params = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    # Compute the inverse perspective transformation matrix
    inv_M = cv2.getPerspectiveTransform(img_params, corner_points_array)
    zone_points = zone_points.astype(np.float32)

    # Map the danger zone back to the camera view
    camera_points = cv2.perspectiveTransform(zone_points.reshape(-1, 1, 2), inv_M)
    return camera_points


##########################################################################################
def xyxy_to_xywh(*xyxy):
    """" Calculates the relative bounding box from absolute pixel values. """
    bbox_left = min([xyxy[0], xyxy[2]])
    bbox_top = min([xyxy[1], xyxy[3]])
    bbox_w = abs(xyxy[0] - xyxy[2])
    bbox_h = abs(xyxy[1] - xyxy[3])
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top + bbox_h / 2)
    w = bbox_w
    h = bbox_h
    return x_c, y_c, w, h


def get_centroids_and_groundpoints(array_boxes_detected):
    """
	For every bounding box, compute the centroid and the point located on the bottom center of the box
	@ array_boxes_detected : list containing all our bounding boxes
	"""
    array_centroids, array_groundpoints = [], []  # Initialize empty centroid and ground point lists
    for index, box in enumerate(array_boxes_detected):
        # Draw the bounding box
        # c
        # Get the both important points
        centroid, ground_point = get_points_from_box(box)
        centroid = centroid[::-1]
        ground_point = ground_point[::-1]
        array_centroids.append(centroid)
        array_groundpoints.append(centroid)
    return array_centroids, array_groundpoints


def get_points_from_box(box):
    """
	Get the center of the bounding and the point "on the ground"
	@ param = box : 2 points representing the bounding box
	@ return = centroid (x1,y1) and ground point (x2,y2)
	"""
    # Center of the box x = (x1+x2)/2 et y = (y1+y2)/2
    center_x = int(((box[1] + box[3]) / 2))
    center_y = int(((box[0] + box[2]) / 2))
    # Coordiniate on the point at the bottom center of the box
    center_y_ground = center_y + ((box[2] - box[0]) / 2)
    return (center_x, center_y), (center_x, int(center_y_ground))


def change_color_on_topview(point, bird_view_img):
    """
    Draw a red circle for the designated point
    """
    cv2.circle(bird_view_img, (int(point[0]), int(point[1])), BIG_CIRCLE, COLOR_RED, 2)
    cv2.circle(bird_view_img, (int(point[0]), int(point[1])), SMALL_CIRCLE, COLOR_RED, -1)


def draw_rectangle(corner_points, frame):
    # Draw rectangle box over the delimitation area
    cv2.line(frame, (corner_points[0][0], corner_points[0][1]), (corner_points[1][0], corner_points[1][1]), COLOR_BLUE,
             thickness=1)
    cv2.line(frame, (corner_points[1][0], corner_points[1][1]), (corner_points[3][0], corner_points[3][1]), COLOR_BLUE,
             thickness=1)
    cv2.line(frame, (corner_points[0][0], corner_points[0][1]), (corner_points[2][0], corner_points[2][1]), COLOR_BLUE,
             thickness=1)
    cv2.line(frame, (corner_points[3][0], corner_points[3][1]), (corner_points[2][0], corner_points[2][1]), COLOR_BLUE,
             thickness=1)


def get_human_box_detection(boxes, scores, classes, height, width):
    """
	For each object detected, check if it is a human and if the confidence >> our threshold.
	Return 2 coordonates necessary to build the box.
	@ boxes : all our boxes coordinates
	@ scores : confidence score on how good the prediction is -> between 0 & 1
	@ classes : the class of the detected object ( 1 for human )
	@ height : of the image -> to get the real pixel value
	@ width : of the image -> to get the real pixel value
	"""
    array_boxes = list()  # Create an empty list
    for i in range(boxes.shape[1]):
        # If the class of the detected object is 1 and the confidence of the prediction is > 0.6
        if int(classes[i]) == 1 and scores[i] > 0.75:
            # Multiply the X coordonnate by the height of the image and the Y coordonate by the width
            # To transform the box value into pixel coordonate values.
            box = [boxes[0, i, 0], boxes[0, i, 1], boxes[0, i, 2], boxes[0, i, 3]] * np.array(
                [height, width, height, width])
            # Add the results converted to int
            array_boxes.append((int(box[0]), int(box[1]), int(box[2]), int(box[3])))
    return array_boxes
