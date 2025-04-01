# Tutorial #6
# -----------
#
# Playing around with colors. We convert some values from RGB to HSV and then find colored objects in the image and mask
# them out. Includes a color picker on double-click now. The RGB version is meant to demonstrate that this does not work
# in RGB color space.

import numpy as np
import cv2

# Print keyboard usage
print("This is a HSV color detection demo. Use the keys to adjust the \
selection color in HSV space. Circle in bottom left.")
print("The masked image shows only the pixels with the given HSV color within \
a given range.")
print("Use h/H to de-/increase the hue.")
print("Use s/S to de-/increase the saturation.")
print("Use v/V to de-/increase the (brightness) value.\n")
print("Double-click an image pixel to select its color for masking.")

# Capture webcam image
cap = cv2.VideoCapture(0)

# Get camera image parameters from get()
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
codec = int(cap.get(cv2.CAP_PROP_CODEC_PIXEL_FORMAT))

print("Video properties:")
print("  Width = " + str(width))
print("  Height = " + str(height))
print("  Codec = " + str(codec))

# Drawing helper variables
thick = 10
thin = 3
thinner = 2
font_size_large = 3
font_size_small = 1
font_size_smaller = 0.6
font = cv2.FONT_HERSHEY_SIMPLEX

# TODO Define RGB colors as variables
red = (255, 0,0) #blue in cv2
green = (0, 255, 0) 
blue =  (0, 0, 255) #red in cv2

# Exemplary color conversion (only for the class), tests usage of cv2.cvtColor
bgr_color_blue = np.array([[red]], dtype=np.uint8)
bgr_color_green = np.array([[green]], dtype=np.uint8)
bgr_color_red = np.array([[blue]], dtype=np.uint8)

hsv_color_blue = cv2.cvtColor(bgr_color_blue, cv2.COLOR_BGR2HSV)
hsv_color_green = cv2.cvtColor(bgr_color_green, cv2.COLOR_BGR2HSV)
hsv_color_red = cv2.cvtColor(bgr_color_red, cv2.COLOR_BGR2HSV)

print(f"Red in RGB and HSV:\n{red}\n{hsv_color_red}")
print(f"Green in RGB and HSV:\n{green}\n{hsv_color_green}")
print(f"Blue in RGB and HSV:\n{blue}\n{hsv_color_blue}")

# TODO Enter some default values and uncomment
hue = 100
hue_range = 10
saturation = 200
saturation_range = 100
value = 200
value_range = 100


# Callback to pick the color on double click
def color_picker(event, x, y, flags, param):
    global hue, saturation, value
    if event == cv2.EVENT_LBUTTONDBLCLK:
        (h, s, v) = hsv[y, x]
        hue = int(h)
        saturation = int(s)
        value = int(v)
        print("New color selected:", (hue, saturation, value))

cv2.namedWindow("original")
cv2.setMouseCallback("original", color_picker)
title_masked_window = "Masked image"
cv2.namedWindow(title_masked_window)
title_mask_window = "Mask image"
cv2.namedWindow(title_mask_window)

while True:
    # Get video frame (always BGR format!)
    ret, frame = cap.read()
    if ret:
        # Copy image to draw on
        img = cv2.flip(frame.copy(), 1)

        # TODO Compute color ranges for display
        lower_color = np.array([hue - hue_range, saturation - saturation_range, value - value_range])
        upper_color = np.array([hue + hue_range, saturation + saturation_range, value + value_range])

        selection_hsv = np.full((1, 1, 3), [hue, saturation, value], dtype=np.uint8)
        selection_bgr = cv2.cvtColor(selection_hsv, cv2.COLOR_HSV2BGR)

        # TODO Draw selection color circle and text for HSV values
        img = cv2.circle(img, (width - 55, height - 55), 50, (selection_bgr[0][0][0], selection_bgr[0][0][1], selection_bgr[0][0][2]), cv2.FILLED)
        img = cv2.putText(img, f"R: {selection_bgr[0][0][2]}", (width - 220, height - 55), font, font_size_small, (255, 255, 255))
        img = cv2.putText(img, f"G: {selection_bgr[0][0][1]}", (width - 220, height - 30), font, font_size_small, (255, 255, 255))
        img = cv2.putText(img, f"B: {selection_bgr[0][0][0]}", (width - 220, height - 5), font, font_size_small, (255, 255, 255))

        # TODO Convert to HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # TODO Create a bitwise mask
        mask = cv2.inRange(hsv, lower_color, upper_color)
        # TODO Apply mask
        masked = cv2.bitwise_and(img, img, mask=mask)

        # TODO Show the original image with drawings in one window
        cv2.imshow("original", img)
        cv2.moveWindow("original", 0, 0)

        # TODO Show the masked image in another window
        cv2.imshow(title_masked_window, masked)
        # cv2.moveWindow(title_masked_window, width / height)

        # TODO Show the mask image in another window
        cv2.imshow(title_mask_window, mask)
        # cv2.moveWindow(title_mask_window, width, int(-height / 2))

        # TODO Deal with keyboard input
        key = cv2.waitKey(10)
        if key == ord("q"):
            break
        if key == ord("h"):
            hue -= 1
        if key == ord("H"):
            hue += 1
        if key == ord("s"):
            saturation -= 1
        if key == ord("S"):
            saturation += 1
        if key == ord("v"):
            value -= 1
        if key == ord("V"):
            value += 1
    else:
        print("Could not start video camera")
        break

cap.release()
cv2.destroyAllWindows()
