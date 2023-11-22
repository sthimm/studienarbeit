import cv2
import os

from config_charuco import ChArucoConfig
from utils import get_charuco_board, get_aruco_dict

# Load configuration from YAML
CONFIG = ChArucoConfig('config_charuco.yaml')
CONFIG.load()

# Obtain ArUco dictionary and Charuco board
ARUCO_DICT = get_aruco_dict(CONFIG.board_type)
BOARD = get_charuco_board(CONFIG)

# Set the size of the board image
size = (CONFIG.size_x, CONFIG.size_y)
# size = (500, 500)

# Draw the Charuco board image
img = BOARD.draw(outSize=size)

# Directory to store the board images
board_directory = 'boards'
os.makedirs(board_directory, exist_ok=True)

# Create a description for the board image filename
img_description = f'{CONFIG.board_type}_{CONFIG.size_x}x{CONFIG.size_y}'
img_path = os.path.join(board_directory, os.path.basename(img_description) + ".png")

# Save the board image
cv2.imwrite(img_path, img)

# Print messages
print('Charuco board image generated and saved.')
print('Image path:', img_path)

# Display the generated board image
cv2.imshow('Generated Charuco Board', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
