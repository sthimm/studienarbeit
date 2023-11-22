import cv2
import numpy as np

from inference import solve_pnp, infer_image
from dc_utils import draw_inner_corners


def cv2_detect(img, dictionary, board, calib_path):

    mtx, dist = np.load(calib_path, allow_pickle=True)

    # Convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect markers
    parameters = cv2.aruco.DetectorParameters_create()
    corners, ids, _ = cv2.aruco.detectMarkers(gray, dictionary, parameters=parameters)
    corners, ids, _, _ = cv2.aruco.refineDetectedMarkers(gray, board, corners, ids, np.array([]))

    # marker_img = cv2.aruco.drawDetectedMarkers(img, corners, ids)
    # cv2.imwrite("images/marker_img.png", marker_img)
    # cv2.imshow("Marker", marker_img)
    # cv2.waitKey(0)

    response, corners, ids = cv2.aruco.interpolateCornersCharuco(
        markerCorners=corners,
        markerIds=ids,
        image=gray,
        board=board
    )

    tvec = np.zeros((1, 3))
    rvec = np.zeros((1, 3))
    ret, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(corners, ids, board, mtx, dist, rvec, tvec)

    if ret:
        img = cv2.drawFrameAxes(img, mtx, dist, rvec, tvec, length=0.01, thickness=3)

        img = draw_inner_corners(img, corners.reshape(-1, 2), ids[:, 0], radius=3, draw_ids=True)

    # cv2.imwrite("images/charuco_position.png", img)
    # cv2.imshow("Charuco", img)
    # cv2.waitKey(0)

    return img, ids, (rvec, tvec)


def dc_detect(img, config, calib_path, deepc, refinenet):

    mtx, dist = np.load(calib_path, allow_pickle=True)

    n_ids = (config.row_count-1)*(config.col_count-1)

    keypoints, img, ids = infer_image(img, n_ids, deepc, refinenet, draw_pred=True)
    ret, rvec, tvec = solve_pnp(keypoints, config.col_count, config.row_count, config.square_len, mtx, dist)

    if ret:
        img = cv2.drawFrameAxes(img, mtx, dist, rvec, tvec, length=0.01, thickness=3)

    return img, ids, (rvec, tvec)
