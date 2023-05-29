import cv2
from PIL import Image


def read_image(full_path, grayscale: bool = False):
    if grayscale:
        return cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.imread(full_path)
    if img is None:
        img = Image.open(full_path)
    return img


def save_image(destination, img_data):
    cv2.imwrite(destination, img_data)


def resize_image(img, new_size):
    return cv2.resize(img, new_size)
