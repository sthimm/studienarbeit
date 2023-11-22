import cv2
import numpy as np
import random


def motion_blur(img, kernel_size):

    # Create the kernel
    kernel = np.zeros((kernel_size, kernel_size))

    # Fill the middle row with ones
    kernel[:, int((kernel_size - 1)/2)] = np.ones(kernel_size)

    # Normalize
    kernel /= kernel_size

    # Apply kernel
    img = cv2.filter2D(img, -1, kernel)

    return img


def change_brightness(img, value):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = cv2.add(v, value)
    v[v > 255] = 255
    v[v < 0] = 0
    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

    return img


def gaussian_blur(img, kernel_size=7):

    blurred_img = cv2.GaussianBlur(img, (kernel_size, kernel_size), cv2.BORDER_DEFAULT)

    return blurred_img


def sp_noise(image, prob):

    output = image.copy()
    if len(image.shape) == 2:
        black = 0
        white = 255
    else:
        colorspace = image.shape[2]
        if colorspace == 3:  # RGB
            black = np.array([0, 0, 0], dtype='uint8')
            white = np.array([255, 255, 255], dtype='uint8')
        else:  # RGBA
            black = np.array([0, 0, 0, 255], dtype='uint8')
            white = np.array([255, 255, 255, 255], dtype='uint8')
    probs = np.random.random(output.shape[:2])
    output[probs < (prob / 2)] = black
    output[probs > 1 - (prob / 2)] = white
    return output


def speckle(image):
    row, col, ch = image.shape
    gauss = np.random.randn(row, col, ch)
    gauss = gauss.reshape(row, col, ch)
    noisy = image + image * gauss
    return noisy

