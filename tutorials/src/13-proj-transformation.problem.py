# Exercise #13
# ------------
#
# Select four points in two images and compute the appropriate projective/perspective transformation. Copy from
# 12-affine-transformation.solution.py and change three lines...
#
# Inspired by https://www.pyimagesearch.com/2015/03/09/capturing-mouse-click-events-with-python-and-opencv/

# Tutorial #12
# ------------
#
# Click three points in two images and compute the appropriate affine transformation. Inspired by
# https://www.pyimagesearch.com/2015/03/09/capturing-mouse-click-events-with-python-and-opencv/

import numpy as np
import cv2

# Define global arrays for the clicked (reference) points
ref_pt_src = []
ref_pt_dst = []

#title
org_title = "Original"
cln_title = "Affine Transform"

# TODO Define one callback functions for each image window
def click_src(event, x, y, flags, param):
    # Grab references to the global variables
    global ref_pt_src
    # If the left mouse button was clicked, add the point to the source array
    if event == cv2.EVENT_LBUTTONDOWN: 
        ref_pt_src.append((x,y))
    # in if block: Draw a circle around the clicked point
        cv2.circle(img, (x,y), 2, (0, 255, 0), 2)
    # in if block: Redraw the image
        cv2.imshow(org_title, img)


def click_dst(event, x, y, flags, param):
    # Grab references to the global variables
    global ref_pt_dst
    # If the left mouse button was clicked, add the point to the source array
    if event == cv2.EVENT_LBUTTONDOWN:
        ref_pt_dst.append((x,y))
    # in if block: Draw a circle around the clicked point
        cv2.circle(clone, (x,y), 2, (0,255,0), 2)
    # in if block: Redraw the image
        cv2.imshow(cln_title, clone)


# Load image and resize for better display
img = cv2.imread('./tutorials/data/images/nl_clown.jpg', cv2.IMREAD_COLOR)
img = cv2.resize(img, (400, 400), interpolation=cv2.INTER_CUBIC)

# Helper variables and image clone for reset
rows, cols, dim = img.shape
clone = img.copy()
dst_transform = np.zeros(img.shape, np.uint8)
# TODO Initialize windows including mouse callbacks

cv2.namedWindow(org_title)
cv2.setMouseCallback(org_title, click_src)
cv2.namedWindow(cln_title)
cv2.setMouseCallback(cln_title, click_dst)

# Keep looping until the 'q' key is pressed
computationDone = False
while True:

    # TODO Change the condition to check if there are three reference points clicked
    if not (computationDone) and len(ref_pt_src) == 4 and len(ref_pt_dst) == 4:
        # TODO Compute the transformation matrix (using cv2.getAffineTransform)
            # T_affine_transform = cv2.getAffineTransform(np.float32(ref_pt_src), np.float32(ref_pt_dst))
            T_projective_transform = cv2.getPerspectiveTransform(np.float32(ref_pt_src), np.float32(ref_pt_dst))
        # TODO print its values
            print("\nMatrix:\n", "\n".join(["\t".join(["%03.3f" % cell for cell in row]) for row in T_projective_transform]))
        # TODO and apply it with cv2.warpAffine
            dst_transform = cv2.warpPerspective(clone, T_projective_transform, (cols, rows))
            computationDone = True
    # TODO Display the image and wait for a keypress
    cv2.imshow(org_title, img)
    cv2.imshow(cln_title, dst_transform)
        # TODO If the 'r' key is pressed, reset the transformation
    key = cv2.waitKey(10)
    if key == ord('r'):
        dst_transform = np.zeros(img.shape, np.uint8)
        img = clone.copy()
        ref_pt_src = []
        ref_pt_dst = []
        computationDone = False
    # TODO If the 'q' key is pressed, break from the loop
    elif key == ord("q"):
        break

cv2.destroyAllWindows()
