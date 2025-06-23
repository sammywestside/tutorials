# Tutorial #20
# ------------
#
# Image classification with k-nearest neighbor approach using the CIFAR-10 data. Code is similar to the one used in
# assignment #4 and hence a bit cluttered.

# TODO Image classification with kNN on CIFAR-10 data set. Go to https://www.cs.toronto.edu/~kriz/cifar.html and
# download the data CIFAR-10 python version: https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz and then
# extract into cifar folder.

import numpy as np
import cv2
from enum import Enum
import pickle
import glob

window_title = "test image"


class Descriptor(Enum):
    """
    Define available descriptors
    """

    COLOR32 = range(1)

    def compute(self, img):
        """
        Compute the descriptor vector
        """

        if self is Descriptor.COLOR32:
            # The descriptor vector is simply the resized image (32x32) in row-major order with
            # each color channel one after one
            return np.ravel(cv2.resize(cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE), (32, 32)), "F")

    def testCompute(self, img):
        """
        Visually test the computation of the descriptor vector
        """

        arr = self.compute(img)

        # Get only the red channel (first 1024 values) and display it as greyscale image
        test_img = np.reshape(arr[:1024], (32, 32))
        cv2.imshow(window_title, test_img)
        cv2.waitKey(0)

    def getSize(self):
        """qq
        Get the length of the descriptor vector
        """

        if self is Descriptor.COLOR32:
            return 32 * 32 * 3


class TrainingSet:

    def __init__(self, root_path="") -> None:
        """
        Create and manage the image data set used as training data
        """

        self.root_path = root_path

    def loadCifar10(self, batch_file, meta_file):
        """
        Load the CIFAR-10 data set and convert structure so that it can be used with OpenCV kNN algorithms
        """

        with open(meta_file, "rb") as fo:
            meta_data = pickle.load(fo, encoding="bytes")

        self.categories = meta_data[b"label_names"]
        self.num_training_images = meta_data[b"num_cases_per_batch"]

        with open(batch_file, "rb") as fo:
            dict = pickle.load(fo, encoding="bytes")

        self.descriptor = Descriptor.COLOR32

        tempData = dict[b"data"]
        tempResponses = np.array(dict[b"labels"])
        self.trainData = np.ndarray(
            shape=(self.num_training_images, self.descriptor.getSize()),
            buffer=np.float32(np.array(np.ravel(tempData))),
            dtype=np.float32,
        )
        self.responses = np.ndarray(shape=(self.num_training_images, 1),
                                    buffer=np.float32(np.ravel(tempResponses)),
                                    dtype=np.float32)
        self.img_files = dict[b"filenames"]

        print("Loaded CIFAR-10 training data:")
        print("Number of images: ", self.num_training_images)
        print("Used descriptor:", self.descriptor)
        print("Categories:", list(map(bytes.decode, self.categories)))

    def testCifar(self, index):
        """
        Test visually if the data is loaded correctly
        """

        test_img = np.uint8(np.reshape(self.trainData[index, 0:1024], (32, 32)))
        cv2.imshow(window_title, test_img)
        cv2.waitKey(0)


def classify_knn(trainData, sample, k):
    """
    Returns the dominating category in the k nearest neighbors
    """

    knn = cv2.ml.KNearest_create()
    knn.train(trainData.trainData, cv2.ml.ROW_SAMPLE, trainData.responses)
    ret, results, neighbours, dist = knn.findNearest(sample, k)

    return trainData.categories[int(ret)]


def run_test_on_folder(trainData, folder_name, k):
    """
    Loads all images found in the given folder and classifies them
    """

    num_test_images = 0

    for file in glob.glob(folder_name + "/*.jpg"):
        print("Testing file: %s" % file)
        descr = trainData.descriptor

        # Load image
        img = cv2.imread(file, cv2.IMREAD_COLOR)

        # Test if the compute method for the descriptor works as expected, a window should pop up that shows the image
        descr.testCompute(img)

        # Compute the descriptor vector
        newcomer = np.ndarray(shape=(1, descr.getSize()), buffer=np.float32(descr.compute(img)), dtype=np.float32)

        # Classify the image
        category = classify_knn(trainData, newcomer, k)
        print("Classified as %s" % category.decode())
        num_test_images += 1

    print("Found %d test images in %s" % (num_test_images, folder_name))


def main():
    # Initialize the training set and load CIFAR-10 data
    trainData = TrainingSet()
    trainData.loadCifar10("./tutorials/data/cifar/cifar-10-batches-py/data_batch_1",
                          "./tutorials/data/cifar/cifar-10-batches-py/batches.meta")

    # Open a window for visual debugging
    cv2.namedWindow(window_title, cv2.WINDOW_GUI_NORMAL)

    # Check if training data is loaded correctly
    trainData.testCifar(375)

    # Test a bunch of images and define an appropriate number of neighbours (k)
    run_test_on_folder(trainData, "./tutorials/data/cifar/", k=10)


if __name__ == "__main__":
    main()
