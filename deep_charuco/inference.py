import cv2
from dataclasses import replace
import numpy as np
import torch

from aruco_utils import draw_inner_corners, label_to_keypoints, cv2_aruco_detect
from typing import Optional
import configs
from configs import load_configuration
from models.model_utils import pred_to_keypoints, extract_patches, pre_bgr_image
from models.net import lModel, dcModel
from models.refinenet import RefineNet, lRefineNet


def solve_pnp(keypoints, col_count, row_count, square_len, camera_matrix, dist_coeffs):
    if keypoints.shape[0] < 4:
        return False, None, None

    # Create inner corners board points
    inn_rc = np.arange(1, row_count)
    inn_cc = np.arange(1, col_count)
    object_points = np.zeros(((col_count - 1) * (row_count - 1), 3), np.float32)
    object_points[:, :2] = np.array(np.meshgrid(inn_rc, inn_cc)).reshape((2, -1)).T * square_len

    image_points = keypoints[:, :2].astype(np.float32)
    object_points_found = object_points[keypoints[:, 2].astype(int)]

    ret, rvec, tvec = cv2.solvePnP(object_points_found, image_points, camera_matrix, dist_coeffs)
    return ret, rvec, tvec


def infer_image(img: np.ndarray, dust_bin_ids: int, deepc: lModel,
                refinenet: Optional[lRefineNet] = None,
                draw_pred: bool = False):
    """
    Do full inference on a BGR image
    """

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray = pre_bgr_image(img_gray, is_gray=True)
    loc_hat, ids_hat = deepc.infer_image(img_gray, preprocessing=False)
    kpts_hat, ids_found = pred_to_keypoints(loc_hat, ids_hat, dust_bin_ids)

    # Draw predictions in RED
    if draw_pred:
        img = draw_inner_corners(img, kpts_hat, ids_found, radius=3,
                                 draw_ids=True, color=(0, 0, 255))

    if ids_found.shape[0] == 0:
        return np.array([]), img

    if refinenet is not None:
        patches = extract_patches(img_gray, kpts_hat)

        # Extract 8x refined corners (in original resolution)
        refined_kpts, _ = refinenet.infer_patches(patches, kpts_hat)

        # Draw refinenet refined corners in yellow
        if draw_pred:
            img = draw_inner_corners(img, refined_kpts, ids_found,
                                     draw_ids=False, radius=1, color=(0, 255, 255))

    keypoints = refined_kpts if refinenet else kpts_hat
    keypoints = np.array([[k[0], k[1], idx] for k, idx in sorted(zip(keypoints,
                                                                     ids_found),
                                                                 key=lambda x:
                                                                 x[1])])
    return keypoints, img


def load_models(deepc_ckpt: str, refinenet_ckpt: Optional[str] = None, n_ids: int = 16, device='cuda'):
    deepc = lModel.load_from_checkpoint(deepc_ckpt, dcModel=dcModel(n_ids))
    deepc.eval()
    deepc.to(device)

    refinenet = None
    if refinenet_ckpt is not None:
        refinenet = lRefineNet.load_from_checkpoint(refinenet_ckpt, refinenet=RefineNet())
        refinenet.eval()
        refinenet.to(device)

    return deepc, refinenet
