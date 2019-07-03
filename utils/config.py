"""
    These are global config options used in a number of files.
    Careful with paths using ~ for home -- os module does not like this
"""

IMAGE_HEIGHT = 480
IMAGE_WIDTH = 752
IMAGE_CHANNELS = 1

IMAGE_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)
RESIZE_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH)

BASE_DIR = "/path/to/current/dir/"
TRAIN_IMAGE_DIR = BASE_DIR + "labelled_data/train/image/"
TRAIN_LABEL_DIR = BASE_DIR + "labelled_data/train/label/"
VALID_IMAGE_DIR = BASE_DIR + "labelled_data/valid/image/"
VALID_LABEL_DIR = BASE_DIR + "labelled_data/valid/label/"

LEARNING_RATE = .0001

SEED = 98

DEQUE_MAXLEN = 256
