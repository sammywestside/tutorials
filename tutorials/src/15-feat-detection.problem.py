# Exercise #15
# ------------
#
# Compute the features of an image with the Harris corner detection. Adjust the parameters using sliders.

import numpy as np
import cv2

file_name = "./tutorials/data/images/logo.png"
window_name = "Feat-Detection"
# TODO Define a function that detects and draws corners into the image
def mark_corners(img, corners):
# Drawing helper variables
    thickness = 2
    radius = 4

# Get a different color array for each of the features/corners
    for i in corners:
        rng = np.random.default_rng()
        color = rng.uniform(low=0.0, high=255.0, size=3)
# Draw a circle around each corner
        x, y = i.ravel()
        img = cv2.circle(img, (x,y), radius, color, thickness)
# Show the resulting image
    cv2.imshow(window_name, img)
# TODO Define the callback function
# Read paremeters from slider positions
def onChange(value):
# Run corner detection
    max_corners = cv2.getTrackbarPos("max_corners: ", window_name)
    # quality_level = cv2.getTrackbarPos("quality_Level: ", window_name)
    # min_distance = cv2.getTrackbarPos("minDistance: ", window_name)

    corners = cv2.goodFeaturesToTrack(gray, max_corners, 0.01, 10)
# cv2.goodFeaturesToTrack returns corners as floating point values,
# hence convert to integer
    corners = np.int8(corners)
# Call the function from above to draw the corners into the image
    mark_corners(img, corners)
    return

# TODO Load example image as color image
img = cv2.imread(file_name, cv2.IMREAD_COLOR)
# TODO Clone image
clone = np.copy(img)
# TODO Create a greyscale image for the corner detection
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# TODO Create a window with sliders and show resulting image
cv2.imshow(window_name, img)
# TODO Create sliders for all parameters and one callback function
cv2.createTrackbar("max_corners: ", window_name, 10, 500, onChange)
# cv2.createTrackbar("quality_Level: ", window_name, 0, 10, onChange)
# cv2.createTrackbar("minDistance: ", window_name, 1, 50, onChange)
# Wait until a key is pressed and end the application
cv2.waitKey(0)
cv2.destroyAllWindows()
