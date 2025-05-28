# Tutorial #17
# ------------
#
# A demonstration of the OpenCV Simple Blob Detector. Adjust the parameters using sliders. The following OpenCV
# functions are meant to be used in this tutorial:
#
# SimpleBlobDetector (with params) see https://docs.opencv.org/3.4/d0/d7a/classcv_1_1SimpleBlobDetector.html#details
# drawKeypoints see https://docs.opencv.org/3.4/d4/d5d/group__features2d__draw.html#gab958f8900dd10f14316521c149a60433

import cv2
import numpy as np

# Setup SimpleBlobDetector parameters.
params = cv2.SimpleBlobDetector_Params()

def detect_draw_and_show_blobs(img):
    # Determine the used global variables
    global params, window_name

    # Create a detector with the parameters
    detector = cv2.SimpleBlobDetector_create(params)

    # Create a greyscale image for the corner detection
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect blobs
    keypoints = detector.detect(img_gray)

    # Use an image clone to for drawing
    img_clone = np.copy(img)

    # Draw detected blobs as blue circles
    # Note that cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size
    # of the circle corresponds to the size of blob.
    img_clone = cv2.drawKeypoints(img_clone, keypoints, np.array([]), (255, 0, 0),
                                  cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Display the image
    cv2.imshow(window_name, img_clone)


# Define the callback function
def on_change(val):
    global params, img

    # Change thresholds
    params.minThreshold = cv2.getTrackbarPos("minThreshold", window_name)
    params.maxThreshold = cv2.getTrackbarPos("maxThreshold", window_name)

    # Filter by Area.
    filter_area_val = cv2.getTrackbarPos("minArea", window_name)

    if filter_area_val > 0:
        params.filterByArea = True
    else:
        params.filterByArea = False

    params.minArea = filter_area_val

    # Filter by Circularity
    filter_circularity_val = cv2.getTrackbarPos("minCircularity (in %%)", window_name)

    if filter_circularity_val > 0:
        params.filterByCircularity = True
    else:
        params.filterByCircularity = False
        filter_circularity_val = 0.1 # OpenCV does not allow 0.0

    print(filter_circularity_val / 100.0)

    params.minCircularity = filter_circularity_val / 100.0

    # Filter by Convexity
    filter_convexity_val = cv2.getTrackbarPos("minConvexity (in %%)", window_name)

    if filter_convexity_val > 0:
        params.filterByConvexity = True
    else:
        params.filterByConvexity = False
        filter_convexity_val = 0.1 # OpenCV does not allow 0.0

    params.minConvexity = filter_convexity_val / 100.0

    # Filter by Inertia
    filter_inertia_val = cv2.getTrackbarPos("minInertiaRatio (in %%)", window_name)

    if filter_inertia_val > 0:
        params.filterByInertia = True
    else:
        params.filterByInertia = False
        filter_inertia_val = 0.1 # OpenCV does not allow 0.0

    params.minInertiaRatio = filter_inertia_val / 100.0

    detect_draw_and_show_blobs(img)


# Load example image as color image
img = cv2.imread("./tutorials/data/images/blobtest.jpg", cv2.IMREAD_COLOR)

# Create a window with sliders and show resulting image
window_name = "Blob detection demo"
cv2.namedWindow(window_name, cv2.WINDOW_GUI_EXPANDED)

# Create sliders for all parameters and one callback function
cv2.createTrackbar("minArea", window_name, 1, 2500, on_change)
cv2.createTrackbar("minThreshold", window_name, 10, 255, on_change)
cv2.createTrackbar("maxThreshold", window_name, 200, 255, on_change)
cv2.createTrackbar("minCircularity (in %%)", window_name, 0, 100, on_change)
cv2.createTrackbar("minInertiaRatio (in %%)", window_name, 0, 100, on_change)
cv2.createTrackbar("minConvexity (in %%)", window_name, 0, 100, on_change)

# Detect, draw and show blobs
detect_draw_and_show_blobs(img)

# Wait until a key is pressed and end the application
cv2.waitKey(0)
cv2.destroyAllWindows()
