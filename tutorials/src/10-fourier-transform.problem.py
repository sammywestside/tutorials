# Tutorial #10
# ------------
#
# Doing the Fourier Transform for images and back. This code is based on the stackoverflow answer from Fred Weinhaus:
# https://stackoverflow.com/a/59995542

import cv2
import numpy as np

# Global helper variables
window_width = 640
window_height = 480
kernel_size = 5
sigma = 0.3*((kernel_size-1)*0.5 - 1) + 0.8


def dilatation(img, size, shape):
    kernel = cv2.getStructuringElement(
        shape, (2 * size + 1, 2 * size + 1), (size, size))
    return cv2.dilate(img, kernel)

# Erosion with parameters


def erosion(img, size, shape):
    kernel = cv2.getStructuringElement(
        shape, (2 * size + 1, 2 * size + 1), (size, size))
    return cv2.erode(img, kernel)

# TODO Implement the function get_frequencies(image):
# Convert image to floats and do dft saving as complex output


def get_frequencies(img):
    # img = dilatation(img, kernel_size, cv2.MORPH_ELLIPSE)
    # img = erosion(img, kernel_size, cv2.MORPH_ELLIPSE)
    # img = cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma)

    dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
# Apply shift of origin from upper left corner to center of image
    polar_coordinates = np.fft.fftshift(dft)
    # cv2.circle(polar_coordinates, (320 - 25, 240), 25, (0, 0, 0), 10)
    # cv2.circle(polar_coordinates, (320 + 25, 240), 25, (0, 0, 0), 10)
    # cv2.rectangle(polar_coordinates, (int(window_width / 3), int(window_height / 3)), (int(window_width / 1.5), int(window_height / 1.5)), (0,0,0), 10)
    # cv2.rectangle(polar_coordinates, ((320 - 25), (240 - 10)), ((320 + 25), (int(window_height / 4))), (0, 0, 0), 10)
# Extract magnitude and phase images
    amplitude, phase = cv2.cartToPolar(polar_coordinates[:, :, 0], polar_coordinates[:, :, 1])
# Get spectrum for viewing only
    spectrum = np.log(amplitude) / 30
# Return the resulting image (as well as the magnitude and phase for the inverse)
    # cv2.circle(spectrum, (int(window_height / 2), int(window_width / 2)), 50, (0,0,0), 50)
    return spectrum, amplitude, phase
# TODO Implement the function create_from_spectrum():
# Convert magnitude and phase into cartesian real and imaginary components


def create_from_spectrum(amplitude, phase):
    # mag = cv2.pow(amplitude, 1.1)
    real, imag = cv2.polarToCart(amplitude, phase)
# Combine cartesian components into one complex image
    original = cv2.merge([real, imag])
# Shift origin from center to upper left corner
    original_ishift = np.fft.ifftshift(original)
# Do idft saving as complex output
    new_img = cv2.idft(original_ishift)
# Combine complex components into original image again
    new_img = cv2.magnitude(new_img[:, :, 0], new_img[:, :, 1])
# Re-normalize to 8-bits
    min, max = np.amin(new_img, (0, 1)), np.amax(new_img, (0, 1))
    print(min, max)
    new_img = cv2.normalize(new_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return new_img

# We use a main function this time: see https://realpython.com/python-main-function/ why it makes sense


def main():
    # Load an image, compute frequency domain image from it and display both or vice versa
    image_name = "./tutorials/data/images/logo.png"

    # Load the image.
    image = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (window_width, window_height))

    # Show the original image
    # Note that window parameters have no effect on MacOS
    title_original = "Original image"
    cv2.namedWindow(title_original, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title_original, window_width, window_height)
    cv2.moveWindow(title_original, 0, 0)
    cv2.imshow(title_original, image)

    result, amplitude, phase = get_frequencies(image)
    # result = np.zeros((window_height, window_width), np.uint8)

    # Show the resulting image
    # Note that window parameters have no effect on MacOS
    title_result = "Frequencies image"
    cv2.namedWindow(title_result, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title_result, window_width, window_height)
    cv2.moveWindow(title_result, int(window_width / 2), int(window_height / 3))
    cv2.imshow(title_result, result)

    back = create_from_spectrum(amplitude, phase)
    # back = np.zeros((window_height, window_width), np.uint8)

    # And compute image back from frequencies
    # Note that window parameters have no effect on MacOS
    title_back = "Reconstructed image"
    cv2.namedWindow(title_back, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title_back, window_width, window_height)
    cv2.moveWindow(title_back, window_width, window_height)
    cv2.imshow(title_back, back)

    key = cv2.waitKey(0)
    cv2.destroyAllWindows()


# Starting the main function
if __name__ == "__main__":
    main()
