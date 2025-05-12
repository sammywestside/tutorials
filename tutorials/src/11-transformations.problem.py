# Tutorial #11
# ------------
#
# Geometric transformations a.k.a. image warping.

import numpy as np
import cv2

width = 400
height = 400

# Load image and resize for better display
img = cv2.imread("./tutorials/data/images/nl_clown.jpg", cv2.IMREAD_COLOR)
img = cv2.resize(img, (400, 400), interpolation=cv2.INTER_CUBIC)
rows, cols, dims = img.shape

# TODO Define translation matrix for translation about 100 pixels to the right and 50 up
T_translation = np.float32([[1, 0, 100], [0, 1, -50]])

# A pretty print for the matrix:
print("\nTranslation\n", "\n".join(["\t".join(["%03.3f" % cell for cell in row]) for row in T_translation]))

# TODO Apply translation matrix on image using cv2.warpAffine
dst_translation = cv2.warpAffine(img, T_translation, (rows + 100, cols + 50))

# TODO Define anisotropic scaling matrix that stretches to double length horizontally
# and squeezes vertically to the half height
T_anisotropic_scaling = np.float32([[2, 0, 0], [0, 0.5, 0]])

print(
    "\nAnisotropic scaling\n",
    "\n".join(["\t".join(["%03.3f" % cell for cell in row]) for row in T_anisotropic_scaling]),
)

# TODO Apply anisotropic scaling matrix on image using cv2.warpAffine
dst_anisotropic_scaling = cv2.warpAffine(img, T_anisotropic_scaling, (int(cols * 2), rows))

# TODO Define rotation matrix for 45° clockwise rotation
alpha = np.deg2rad(45)
T_rotation = np.float32([[np.cos(alpha), np.sin(alpha), 0], [-np.sin(alpha), np.cos(alpha), 0]])

print("\nRotation\n", "\n".join(["\t".join(["%03.3f" % cell for cell in row]) for row in T_rotation]))

# TODO Apply rotatio matrix on image using cv2.warpAffine
dst_rotation = cv2.warpAffine(img, T_rotation, (cols, rows))

# TODO Rotate around image center for 45° counterclockwise using cv2.getRotationMatrix2D
T_rotation_around_center = cv2.getRotationMatrix2D((cols / 2, rows / 2), 45, 1.0)

print(
    "\nRotation around center\n",
    "\n".join(["\t".join(["%03.3f" % cell for cell in row]) for row in T_rotation_around_center]),
)

# TODO Apply rotatio matrix on image using cv2.warpAffine
dst_rotation_around_center = cv2.warpAffine(img, T_rotation_around_center, (cols, rows))

# Show the original and resulting images
cv2.imshow("Original", img)
cv2.imshow("Translation", dst_translation)
cv2.imshow("Anisotropic scaling", dst_anisotropic_scaling)
cv2.imshow("Rotation", dst_rotation)
cv2.imshow("Rotation around center", dst_rotation_around_center)

# Keep images open until key pressed
cv2.waitKey(0)
cv2.destroyAllWindows()
