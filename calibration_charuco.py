import cv2
import numpy as np
import os

from config_charuco import ChArucoConfig
from utils import get_charuco_board, get_aruco_dict, detect_corners_ids, evaluate_calibration, save_data_npy

# Load configuration from YAML
CONFIG = ChArucoConfig("config_charuco.yaml")
CONFIG.load()

# Obtain ArUco dictionary and Charuco board
ARUCO_DICT = get_aruco_dict(CONFIG.board_type)
BOARD = get_charuco_board(CONFIG)

VIDEO_FILENAME = 'calibration_video_5x5'

# Detect corners and IDs from video frames
print('Detecting Charuco corners and IDs...')
corners_all, ids_all, image_size = detect_corners_ids(
    video_path=os.path.join('input_videos', VIDEO_FILENAME + '.mp4'),
    board=BOARD,
    dict=ARUCO_DICT,
    step=10,
    draw=True
)
print('Charuco corners and IDs detection completed.\n')

# Calibration flags and criteria
FLAGS = (cv2.CALIB_USE_INTRINSIC_GUESS + cv2.CALIB_RATIONAL_MODEL + cv2.CALIB_FIX_ASPECT_RATIO)
CRITERIA = (cv2.TERM_CRITERIA_EPS & cv2.TERM_CRITERIA_COUNT, 10000, 1e-9)

# Initialize camera matrix and distortion coefficients
MTX_INIT = np.array([[1000., 0., image_size[1] / 2],
                     [0., 1000., image_size[0] / 2],
                     [0., 0., 1.]])

DIST_INIT = np.zeros((5, 1))

print('Starting Charuco calibration...')
# Calculate calibration parameters using Charuco calibration
ret, mtx, dist, rot, trans, devIntrinsics, devExtrinsics, perViewErrors = cv2.aruco.calibrateCameraCharucoExtended(
    charucoCorners=corners_all,
    charucoIds=ids_all,
    board=BOARD,
    imageSize=image_size,
    cameraMatrix=MTX_INIT,
    distCoeffs=DIST_INIT,
    flags=FLAGS,
    criteria=CRITERIA
)
print('Charuco calibration completed.\n')

# Evaluate camera calibration with reprojection errors
print('Evaluating camera calibration...\n')
evaluate_calibration(perViewErrors)

# Save calibration data
print('Saving calibration data...')
calib_data = (mtx, dist)
save_data_npy(calib_data, folder='intrinsics', description=VIDEO_FILENAME)
print('Calibration data saved.\n')

# Test undistort of test image
img = cv2.imread("charuco.png")
img_undistorted = cv2.undistort(img, mtx, dist)
cv2.imshow("Undistorted Image", img_undistorted)
cv2.waitKey(0)
cv2.imwrite("undistorted.png", img_undistorted)
