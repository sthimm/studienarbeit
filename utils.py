import cv2
import numpy as np
import os
import torch

from aruco_dict import ARUCO_DICT


def get_charuco_board(config):

    board = cv2.aruco.CharucoBoard_create(
        squaresX=config.col_count,
        squaresY=config.row_count,
        squareLength=config.square_len,
        markerLength=config.marker_len,
        dictionary=get_aruco_dict(config.board_type)
    )

    return board


def get_aruco_dict(board_type):
    return cv2.aruco.Dictionary_get(ARUCO_DICT[board_type])


def detect_corners_ids(video_path, board, dict, step, draw=False):

    # Corners discovered in all images and corresponding ids
    corners_all = []
    ids_all = []

    cap = cv2.VideoCapture(video_path)

    frame_count = 0
    image_size = None

    print("Processing frames")
    print()

    while cap.isOpened():

        ret, frame = cap.read()

        if ret is False:
            break

        frame_count += 1

        if frame_count == step:
            frame_count = 0

            # Grayscale the image
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if image_size is None:
                image_size = gray.shape[::-1]

            # Find aruco markers in the query image
            corners, ids, _ = cv2.aruco.detectMarkers(
                image=gray,
                dictionary=dict
            )

            # If none found, take another capture
            if ids is None:
                continue

            if draw:
                # Outline the aruco markers found in our query image
                frame = cv2.aruco.drawDetectedMarkers(
                    image=frame,
                    corners=corners
                )

            # Interpolate of Charuco Corners
            response, corners, ids = cv2.aruco.interpolateCornersCharuco(
                markerCorners=corners,
                markerIds=ids,
                image=gray,
                board=board
            )

            # Add found corners and ids to calibration arrays
            corners_all.append(corners)
            ids_all.append(ids)

            if draw:
                # Draw detected Charuco board
                frame = cv2.aruco.drawDetectedCornersCharuco(
                    image=frame,
                    charucoCorners=corners,
                    charucoIds=ids
                )

                proportion = max(frame.shape) / 1000.0
                frame = cv2.resize(frame, (int(frame.shape[1] / proportion), int(frame.shape[0] / proportion)))

                cv2.imshow("Detected Charuco board", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    if draw:
        cv2.destroyAllWindows()

    if len(corners_all) == 0:
        print("Detection was unsuccesful")
        exit()

    return corners_all, ids_all, image_size


def evaluate_calibration(reproj_errors):

    mean_error = np.average(reproj_errors)
    print("Average reprojection error: ", mean_error)
    print()


def save_data_npy(data, folder, description):

    os.makedirs(folder, exist_ok=True)
    data_path = os.path.join(folder, os.path.basename(description) + ".npy")
    data = np.array(data, dtype=object)
    np.save(data_path, data)


def adjust_size(img, target_width, target_height):

    img = cv2.pyrDown(img)

    crop_x_start = (img.shape[0] - target_height) // 2
    crop_x_end = crop_x_start + target_height
    crop_y_start = (img.shape[1] - target_width) // 2
    crop_y_end = crop_y_start + target_width

    img = img[crop_x_start:crop_x_end, crop_y_start:crop_y_end]

    return img
