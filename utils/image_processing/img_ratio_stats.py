"""
Calculate aspect ratio between width and height of an image,
then showing frequencies of ratios from a dataset.
As each image has different size/ratio,
it helps determine to best resize factor when applying resizing operator.
"""
import os
from collections import Counter

from config import PALLET_DS_HOME
from utils.image_processing import read_image


def read_folder(folder: str):
    arr = []
    for f in os.listdir(folder):
        path = os.path.join(folder, f)
        img = read_image(path)
        arr.append(round(img.shape[1]/img.shape[0], 2))
    c = Counter(arr)
    print(c.most_common())


# read_folder(f'{PALLET_DS_HOME}/camera-1-light-on/')
