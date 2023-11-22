import cv2
import os

import numpy as np
import torch.cuda

from config_charuco import ChArucoConfig
from pose_estimation import cv2_detect, dc_detect
from utils import get_charuco_board, get_aruco_dict, adjust_size
from inference import load_models
from filter import change_brightness, gaussian_blur, sp_noise, speckle

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

cap = cv2.VideoCapture(VIDEO_PATH)

sum_cv2_ids = 0
sum_dc_ids = 0
num_frames = 0

while cap.isOpened():

    ret, frame = cap.read()

    if ret is False:
        break

    num_frames += 1

    frame = adjust_size(frame, 320, 240)

    frame2 = frame.copy()

    cv2_frame, cv2_ids, cv2_pos = cv2_detect(frame, ARUCO_DICT, BOARD, CALIB_PATH)
    dc_frame, dc_ids, dc_pos = dc_detect(frame2, CONFIG, CALIB_PATH, DEEPC, REFINENET)

    sum_cv2_ids += np.unique(cv2_ids[:, 0].shape[0])
    sum_dc_ids += np.unique(dc_ids.shape[0])

    ref = np.concatenate((cv2_frame, dc_frame), axis=1)

    cv2.imshow("Reference", ref)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("Average number of detected ids with CV2 Library: ", sum_cv2_ids/num_frames)
print("Average number of detected ids with Deep Charuco: ", sum_dc_ids/num_frames)
