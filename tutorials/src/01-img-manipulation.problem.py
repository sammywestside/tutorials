# Exercise #1
# -----------
#
# Load, resize and rotate an image. And display it to the screen.

# TODO First step is to import the opencv module which is called 'cv2'
import cv2

# TODO Check the opencv version
print(f"Opencv version: {cv2.__version__}")

# TODO Load an image with image reading modes using 'imread'
image = cv2.imread("./tutorials/data/images/logo.png")
# cv2.IMREAD_UNCHANGED  - If set, return the loaded image as is (with alpha
#                         channel, otherwise it gets cropped). Ignore EXIF
#                         orientation.
# cv2.IMREAD_GRAYSCALE  - If set, always convert image to the single channel
#                         grayscale image (codec internal conversion).
# cv2.IMREAD_COLOR      - If set, always convert image to the 3 channel BGR
#                         color image.

# TODO Resize image with 'resize'
height = 500
width = 300
image = cv2.resize(image, (height, width))

# TODO Rotate image (but keep it rectangular) with 'rotate'
image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

# TODO Save image with 'imwrite'

# cv2.imwrite("logo_tutorial_1.png", image)

# TODO Show the image with 'imshow'
cv2.imshow("LogoTest", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
