# Exercise #2
# -----------
#
# Direct pixel access and manipulation. Set some pixels to black, copy some part of the image to some other place,
# count the used colors in the image

import cv2
import numpy as np

# TODO Loading images in grey and color
img_path = "./tutorials/data/images/smarties02.jpg"
img_grey = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
img_color = cv2.imread(img_path, cv2.IMREAD_COLOR)

# TODO Do some print out about the loaded data using type, dtype and shape
print(f"Image_Type: {type(img_grey)}\nImage_DType: {img_grey.dtype}\nIMage_shape: {img_grey.shape}\n")
print(f"\nImage_Color_Type: {type(img_color)}\nImage_Color_DType: {img_color.dtype}\nIMage_Color_shape: {img_color.shape}")
# TODO Continue with the grayscale image
# TODO Extract the size or resolution of the image

img_grey_height, img_grey_width = img_grey.shape
img_color_height, img_color_width, img_color_channels = img_color.shape

# TODO Resize image

img_grey_new_height = 7
img_grey_new_width = 5
img_grey = cv2.resize(img_grey, (img_grey_new_height, img_grey_new_width))

# Row and column access, see https://numpy.org/doc/stable/reference/arrays.ndarray.html for general access on ndarrays
# TODO Print first row

# img_array = np.ndarray(img_grey.shape)
# print(img_array[0])

# TODO Print first column

# print(img_array[:, 0])

# TODO Continue with the color image
# TODO Set an area of the image to black

# img_color[100:150, 100:150] = 0

# TODO Show the image and wait until key pressed

# cv2.imshow("Smarties_Color", img_color)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# TODO Find all used colors in the image
colors= cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)
unique_colors = np.unique(colors.reshape(-1, colors.shape[2]), axis=0)
print(unique_colors)
# TODO Copy one part of an image into another one
frame = img_color[100:150, 100:150]
img_color[img_color_height - 50:img_color_height, img_color_width - 50:img_color_width] = frame

# TODO Save image to a file

# cv2.imwrite("smarties_color_tutorial_2.png", img_color)

# TODO Show the image again

# cv2.imshow("Smarties_Grey", img_grey)
cv2.imshow("Smarties_Color", img_color)
cv2.waitKey(0)
cv2.destroyAllWindows()

# TODO Show the original image (copy demo)
