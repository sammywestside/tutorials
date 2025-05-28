# Tutorial #14
# ------------
#
# Compute the edges of an image with the Canny edge detection. Adjust the parameters using sliders.

import numpy as np
import cv2
import math

file_url = "./tutorials/data/images/nl_clown.jpg"
trackbar_one_name = "Blur: "
trackbar_two_name = "T_lower: "
trackbar_three_name = "T_upper: "

def show_images_side_by_side(img_A, img_B):
    """Helper function to draw two images side by side"""
    cv2.imshow(window_name, np.concatenate((img_A, img_B), axis=1))
    return


# TODO: Define callback function
"""callback function for the sliders"""
# Read slider positions
def onChange(value):
    blur_pos = cv2.getTrackbarPos(trackbar_one_name, window_name)
    T_lower = cv2.getTrackbarPos(trackbar_two_name, window_name)
    T_upper = cv2.getTrackbarPos(trackbar_three_name, window_name)
# Blur the image
    blur_pos = int(blur_pos)
    if blur_pos % 2 == 1:
        blur_pos + 1
    
    sigma = 0.3*((blur_pos-1)*0.5 - 1) + 0.8
    img = cv2.GaussianBlur(clone, (blur_pos, blur_pos), sigma)

# Run Canny edge detection with thresholds set by sliders
    edges = cv2.Canny(img, T_lower, T_upper)
# Show the resulting images in one window using the show_images_side_by_side function
    show_images_side_by_side(img, edges)

    return


# TODO Load example image as grayscale
gray = cv2.imread(file_url, cv2.IMREAD_GRAYSCALE)
# Resize if needed
gray = cv2.resize(gray, (600, 400))
# Clone if needed
clone = np.copy(gray)
# TODO Initial Canny edge detection result creation
edges = cv2.Canny(clone, 30, 150)
# TODO Create window with sliders
# Define a window name
window_name = "Canny edge detection demo"
# TODO Show the resulting images in one window
show_images_side_by_side(gray, edges)
# TODO Create trackbars (sliders) for the window and define one callback function
cv2.createTrackbar(trackbar_one_name, window_name, 1, 150, onChange)
cv2.createTrackbar(trackbar_two_name, window_name, 30, 255, onChange)
cv2.createTrackbar(trackbar_three_name, window_name, 240, 255, onChange)

# Wait until a key is pressed and end the application
cv2.waitKey(0)
cv2.destroyAllWindows()
