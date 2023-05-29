"""
convert origin image to grayed version
"""

import os
import pathlib
import random

import cv2

from config import *
from utils.image_processing import read_image, save_image


def test_random_image(dataset: DatasetType = DatasetType.CAMERA_ON, img_type: ImageType = ImageType.JPG):
    """
    randomly select an image and showing its grayed version
    :param img_type: desired format
    :param dataset: datatype with camera on or off
    :return: None
    """
    folder = os.path.join(PALLET_DS_HOME, dataset.value)
    random_img_file = str(random.choice(list(pathlib.Path(folder).glob("*.{}".format(img_type.value)))))

    img = read_image(random_img_file)  # read image data
    img_ratio = img.shape[1] / img.shape[0]  # ratio = width/height
    resized_w = 1280  # resize to fit window
    img = cv2.resize(img, dsize=(resized_w, int(resized_w/img_ratio)), interpolation=cv2.INTER_AREA)
    grayed = to_gray(img)

    cv2.imshow('original', img)
    cv2.imshow('grayed', grayed)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def to_gray(original_img):
    return cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)


def gray_a_folder(inp_folder: str, out_folder: str, max_images: int = None, random_select: bool = None):
    """
    read original (colored) images from a folder (no recursive), gray all images and save to another folder
    :param inp_folder: name of input folder
    :param out_folder: name of destination folder which stores new grayed images
    :param max_images: max number of image will be grayed
    :param random_select:
    :return:
    """
    def is_image_file(file_name):
        _, extension = os.path.splitext(file_name)
        if not isinstance(extension, str):
            return False
        if len(extension) < 1:
            return False
        return extension[1:].lower() in [ImageType.JPG.value, ImageType.BMP.value]

    if inp_folder == out_folder:
        raise Exception('input & output folder can not be the same!')

    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
        print('created new folder', out_folder)
    else:
        raise Exception('Output folder {} is not empty!'.format(out_folder))

    all_files = [
        os.path.join(inp_folder, file)
        for file in os.listdir(inp_folder)
        if is_image_file(file) and os.path.isfile(os.path.join(inp_folder, file))
    ]

    if random_select:
        random.shuffle(all_files)

    if max_images:
        all_files = all_files[:max_images]

    count = 0
    for file in all_files:
        try:
            basename = os.path.basename(file)
            img_data = read_image(file)
            gray_img = to_gray(img_data)
            save_image(os.path.join(out_folder, basename), gray_img)
            count += 1
        except Exception as e:
            print('ERROR - {} - {}'.format(file, e))

    print('saved {} grayed images to {}'.format(count, out_folder))


# test_random_image(DatasetType.CAMERA_ON, ImageType.BMP)

# gray_a_folder(
#     inp_folder=CAM_OFF_RAW_DS,
#     out_folder=CAM_OFF_GRAY_DS_FOLDER,
#     max_images=5000,
#     random_select=False
# )
