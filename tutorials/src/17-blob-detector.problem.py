# Tutorial #17
# ------------
#
# A demonstration of the OpenCV Simple Blob Detector. Adjust the parameters using sliders. The following OpenCV
# functions are meant to be used in this tutorial:
#
# SimpleBlobDetector (with params) see https://docs.opencv.org/4.x/d0/d7a/classcv_1_1SimpleBlobDetector.html#details
# drawKeypoints see https://docs.opencv.org/4.x/d4/d5d/group__features2d__draw.html#gab958f8900dd10f14316521c149a60433

import cv2
import numpy as np

# TODO Setup SimpleBlobDetector parameters
params = cv2.SimpleBlobDetector_Params()
params.filterByColor = False
# TODO Define a function to detect blobs, draw them and display the image
# Determine the used global variables


def detect_blobs(img):
    # Create a detector with the parameters
    detector = cv2.SimpleBlobDetector_create(params)
# Create a greyscale image for the corner detection
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Detect blobs
    blobs = detector.detect(grey)
# Use an image clone to for drawing
    clone = np.copy(img)
# Draw detected blobs as blue circles
# Note that cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size
# Of the circle corresponds to the size of blob.
    output = cv2.drawKeypoints(clone, blobs, np.array([]), (255, 0, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
# Display the image
    cv2.imshow("Blob_Detector", output)
# TODO Define the callback function


def callback(value):
    # Provide access to the global blob detector parameters
    # Use an image clone to for drawing
    # TODO Filter by area
    area_value = cv2.getTrackbarPos("Area", "Blob_Detector")
    if area_value > 0:
        params.filterByArea = True
    else:
        params.filterByArea = False

    params.minArea = area_value
# TODO Change threshold parameters
    # threshold_value = cv2.getTrackbarPos("Threshold", "Blob_Detector")

    # params.minThreshold = 10
    # params.maxThreshold = 200
# TODO Filter by circularity
    circ_value = cv2.getTrackbarPos("Circularity", "Blob_Detector") / 100
    if circ_value > 0:
        params.filterByCircularity = True
    else:
        params.filterByCircularity = False
        circ_value = 0.1

    params.minCircularity = circ_value
# TODO Filter by inertia
    inertia_value = cv2.getTrackbarPos("Inertia", "Blob_Detector") / 100
    if inertia_value > 0:
        params.filterByInertia = True
    else:
        params.filterByInertia = False
        inertia_value = 0.1

    params.minInertiaRatio = inertia_value
# TODO Filter by convexity
    conv_value = cv2.getTrackbarPos("Convexity", "Blob_Detector") / 100
    if conv_value > 0:
        params.filterByConvexity = True
    else:
        params.filterByConvexity = False
        conv_value = 0.1

    params.minConvexity = conv_value
# Call the detect, draw and show function
    detect_blobs(img)


# Load example image as color image
img = cv2.imread("./tutorials/data/images/blobtest.jpg", cv2.IMREAD_COLOR)

# TODO Create a window with sliders and show resulting image
cv2.namedWindow("Blob_Detector", cv2.WINDOW_AUTOSIZE)
# HINT: Create sliders for all parameters using only one callback function
cv2.createTrackbar("Area", "Blob_Detector", 1, 2500, callback)
# cv2.createTrackbar("Threshold", "Blob_Detector", 10, 200, callback)
cv2.createTrackbar("Circularity", "Blob_Detector", 0, 100, callback)
cv2.createTrackbar("Convexity", "Blob_Detector", 0, 100, callback)
cv2.createTrackbar("Inertia", "Blob_Detector", 0, 100, callback)
# TODO Call the detect, draw and show function
detect_blobs(img)
# TODO Wait until a key is pressed and end the application
cv2.waitKey(0)
cv2.destroyAllWindows()
