import cv2
import os
import numpy as np
import torch.cuda

from config_charuco import ChArucoConfig
from pose_estimation import cv2_detect, dc_detect
from utils import get_charuco_board, get_aruco_dict, adjust_size
from inference import load_models
from filter import change_brightness, gaussian_blur, sp_noise, speckle, motion_blur

# Load configuration from YAML
CONFIG = ChArucoConfig('config_charuco.yaml')
CONFIG.load()

# Obtain ArUco dictionary and Charuco board
ARUCO_DICT = get_aruco_dict(CONFIG.board_type)
BOARD = get_charuco_board(CONFIG)

CALIB_FILENAME = 'input_video_5x5'
CALIB_PATH = os.path.join('intrinsics', CALIB_FILENAME + '.npy')

VIDEO_FILENAME = 'distance'
VIDEO_PATH = os.path.join('input_videos', VIDEO_FILENAME + '.mp4')

DEEPC_PATH = r'reference/longrun-epoch=99-step=369700.ckpt'
REFINENET_PATH = r'reference/second-refinenet-epoch-100-step=373k.ckpt'

N_IDS = (CONFIG.row_count-1)*(CONFIG.col_count-1)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DEEPC, REFINENET = load_models(DEEPC_PATH, REFINENET_PATH, N_IDS, DEVICE)

FILTER_NAME = 'brightness'
FILTER_VALUES = [-90, -80, -70, -60, -50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
FRAME_LIMIT = 100
OUTCOME = np.zeros((len(FILTER_VALUES), 2))

for val in FILTER_VALUES:

    print("Analysis conducted for value: ", val)

    # Open new Video cap
    cap = cv2.VideoCapture(VIDEO_PATH)

    # Found ids with cv2 and dc method
    sum_cv2_ids = 0
    sum_dc_ids = 0

    # Frame counter
    frame_counter = 0

    while cap.isOpened():

        ret, frame = cap.read()

        if ret is False:
            break

        frame_counter += 1

        # Break after FRAME_LIMIT
        if frame_counter >= FRAME_LIMIT:
            break

        # Adjust frame size
        frame = adjust_size(frame, 320, 240)
        frame2 = frame.copy()

        # Filter images to test robustness of cv2 solution
        # frame = motion_blur(frame, val)
        # frame2 = motion_blur(frame2, val)
        frame = change_brightness(frame, val)
        frame2 = change_brightness(frame2, val)
        # frame = gaussian_blur(frame, kernel_size=val)
        # frame2 = gaussian_blur(frame, kernel_size=val)
        # frame = sp_noise(frame, val)
        # frame2 = sp_noise(frame, val)

        # Detect Charuco Marker with CV2 method
        cv2_frame, cv2_ids, cv2_pos = cv2_detect(frame, ARUCO_DICT, BOARD, CALIB_PATH)
        # Detect Charuco Marker with Deep Charuco method
        dc_frame, dc_ids, dc_pos = dc_detect(frame2, CONFIG, CALIB_PATH, DEEPC, REFINENET)

        # Delete duplicates and add to sum
        sum_cv2_ids += np.unique(cv2_ids[:, 0].shape[0])
        sum_dc_ids += np.unique(dc_ids.shape[0])

        # Concatenate and show frame
        ref = np.concatenate((cv2_frame, dc_frame), axis=0)
        cv2.imshow("Charuco Detect", ref)   # Left: CV2, Right: Deep Charuco

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    print("Average number of detected ids with CV2 Library: ", sum_cv2_ids/FRAME_LIMIT)
    print("Average number of detected ids with Deep Charuco: ", sum_dc_ids/FRAME_LIMIT)










