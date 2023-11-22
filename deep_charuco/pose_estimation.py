import cv2
import numpy as np
import torch.cuda

from inference import load_models, infer_image, solve_pnp


def inf_single_dc(img, n_ids, deepc, refinenet, col_count, row_count,
                  square_len, camera_matrix, dist_coeffs, draw=False):
    keypoints, out_img = infer_image(img, n_ids, deepc, refinenet, draw_pred=draw)

    ret, rvec, tvec = solve_pnp(keypoints, col_count, row_count, square_len, camera_matrix, dist_coeffs)

    return out_img


def main():

    camera_matrix, distortion_coefficients = np.load('intrinsics/intrinsics_5x5.npy', allow_pickle=True)

    # Configuration
    aruco_dict = 'DICT_4x4_50'
    row_count = 5
    col_count = 5
    square_len = 0.01
    marker_len = 0.0075

    deepc_path = r'C:\Users\sthimm\simon-thimm\source\deep_charuco\reference\longrun-epoch=99-step=369700.ckpt'
    refinenet_path = r'C:\Users\sthimm\simon-thimm\source\deep_charuco\reference\second-refinenet-epoch-100-step=373k.ckpt'
    n_ids = 16
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(str(device))
    deepc, refinenet = load_models(deepc_path, refinenet_path, n_ids, device=device)

    video_path = r'C:\Users\sthimm\simon-thimm\source\deep_charuco\input_video\input_video_5x5_480x480.mp4'
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():

        ret, frame = cap.read()

        frame = cv2.resize(frame, (320, 240))

        output = inf_single_dc(frame, n_ids, deepc, refinenet, col_count, row_count,
                               square_len, camera_matrix, distortion_coefficients, draw=True)

        output = cv2.resize(output, (1920, 1080))

        cv2.imshow("Deep Charuco", output)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    """ Notiz für Montag: Probieren, ob wenn man Bilder mit 320 auf 240 in das Netz reingibt, ob es dann funktioniert. Nicht als Video reingeben !!!!! Siehe Beispiel Bild"""


if __name__ == '__main__':
    main()
