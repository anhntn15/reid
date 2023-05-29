"""
divide an image into multiple equal parts
"""

import os.path
from math import ceil
from typing import List, Tuple

from config import *
from utils.image_processing import read_image, save_image


def get_cutting_anchors(W, H, w, h) -> List[Tuple]:
    """
    anchor point is the top-left point of the rectangle which is cropped from original image
    :param W: width of original image
    :param H: height of original image
    :param w: width of cropped area
    :param h: height of cropped area
    :return: list of anchor points, e.g: [(0, 0), (100, 150)]
    """
    horizontal_parts = ceil(W/w)
    vertical_parts = ceil(H/h)
    row_extra_step = (horizontal_parts*w - W) / (horizontal_parts - 1) if horizontal_parts > 1 else 0
    col_extra_step = (vertical_parts*h - H) / (vertical_parts - 1) if vertical_parts > 1 else 0

    anchor_points = []
    anchor_x, anchor_y = 0, 0

    for i in range(horizontal_parts):
        anchor_y = 0
        for j in range(vertical_parts):
            anchor_points.append((int(anchor_x), int(anchor_y)))
            anchor_y += h - col_extra_step
        anchor_x += w - row_extra_step

    return anchor_points


def crop_image(img_name: str, out_folder: str, crop_size: tuple, grayscale: bool = False):
    """
    crop an image into multiple equal parts
    :param img_name: full path
    :param crop_size: e.g (32, 32)
    :param out_folder: output folder
    :param grayscale: process a gray image
    :return: list of cropped images
    """
    basename, ext = os.path.splitext(os.path.basename(img_name))
    img = read_image(img_name, grayscale)
    w, h = crop_size
    cropped_points = get_cutting_anchors(img.shape[0], img.shape[1], w, h)
    cropped_images = [img[x:x+w, y:y+h] for (x, y) in cropped_points]
    for i, cimg in enumerate(cropped_images):
        if (cimg.shape[0], cimg.shape[1]) == crop_size:
            save_image(os.path.join(out_folder, '{}_{}{}'.format(basename, i, ext)), cimg)


def crop_a_folder(inp_folder: str, out_folder: str, crop_size: tuple, max_images: int = None, grayscale: bool = False):
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
        print('created new folder', out_folder)
    else:
        raise Exception('output folder {} is not empty!'.format(out_folder))

    inp_images = os.listdir(inp_folder)
    if max_images and max_images < len(inp_images):
        inp_images = inp_images[:max_images]

    print('processing {} images ..'.format(len(inp_images)))
    for file in inp_images:
        try:
            crop_image(os.path.join(inp_folder, file), out_folder, crop_size, grayscale)
        except Exception as e:
            print('ERROR', file, e)

    print("done with {} cropped images."
          .format(len([name for name in os.listdir(out_folder) if os.path.isfile(os.path.join(out_folder, name))])))


# print(get_cutting_anchors(100, 100, 40, 40))

# crop_image('/home/anhntn/PycharmProjects/dae/dataset/test/test.bmp',
#            '/home/anhntn/PycharmProjects/dae/dataset/test/',
#            (600, 1000))

# crop_a_folder(
#     inp_folder=CAM_OFF_GRAY_DS_FOLDER,
#     out_folder=CAM_OFF_CROP_GRAY_DS_FOLDER,
#     crop_size=(128, 128),
#     max_images=500,
#     grayscale=True
# )
