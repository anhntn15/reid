import enum
import json
import os.path

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def read_ds_home():
    f = open(os.path.join(PROJECT_ROOT, 'config', 'config.json'))
    x = json.load(f)
    f.close()
    return x['dataset_home']


PALLET_DS_HOME = read_ds_home()
WORKING_DS_HOME = '/data/palette/dataset'
LOGGING_FILE = os.path.join(PROJECT_ROOT, 'tmp', 'debug.log')


if not os.path.exists(os.path.join(PROJECT_ROOT, 'tmp')):
    os.mkdir(os.path.join(PROJECT_ROOT, 'tmp'))


class DatasetType(enum.Enum):
    CAMERA_OFF = 'camera-0-light-off'
    CAMERA_ON = 'camera-1-light-on'


class ImageType(enum.Enum):
    JPG = 'jpg'
    BMP = 'bmp'


CAM_OFF_RAW_DS = os.path.join(PALLET_DS_HOME, DatasetType.CAMERA_OFF.value)
CAM_OFF_GRAY_DS_FOLDER = os.path.join(WORKING_DS_HOME, 'camera-off-grayed')
CAM_OFF_CROP_GRAY_DS_FOLDER = CAM_OFF_GRAY_DS_FOLDER + '-cropped'


def read_config(path: str):
    with open(path, 'r') as f:
        from yaml import safe_load
        return safe_load(f)
