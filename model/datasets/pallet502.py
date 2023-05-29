import json
import os.path
import random
import re

from torch.utils.data import Dataset

from config import PALLET_DS_HOME
from model.datasets import generate_triplet_pairs
from utils.image_processing import read_image
from utils.log_config import get_custom_logger

logger = get_custom_logger(__name__, 'DEBUG')
PID_PATTERN = re.compile(r'cropped_(\d+)_\w')


def get_pid(img_name: str) -> str:
    """
    the outer function handle exception if there is no pattern matched
    :param img_name: handle image name with structure like 'cropped_123_RR.jpg'
    :return: pid pattern, e.g: '123'
    """
    return PID_PATTERN.findall(img_name)[0]


class DatasetConfig:
    """
    instead of fixed train/test/validation set, we would like to randomly select each subset
    (similar idea like cross-validation)

    class manage the train/test splitting jobs:
        + split dataset by image_id or pallet_id
        + keep track which sample is train/test
    """

    def __init__(self, ds_folder, max_sample: int = None,
                 train_ratio: float = 1, split_by_pid: bool = False,
                 shuffle: bool = False, path_to_subsets: str = None, **kwargs):
        """
        read dataset folder, do train/test split
        :param max_sample: maximum number of image to load (instead of load entire folder)
        :param train_ratio: a float number (0, 1) indicates percentage of training dataset
        :param split_by_pid: split train/test by image_id or pallet_id
        :param shuffle: random order of images loading from folder, helpful when max_sample < len(dataset)
        :param path_to_subsets: if present, read train/test/gallery subset from fixed list
        """
        self.train_set, self.test_set, self.gallery_set = [], [], []

        files = os.listdir(ds_folder)
        if path_to_subsets:
            self.read_subsets_from_file(files, path_to_subsets)
            return

        if shuffle:
            random.shuffle(files)
        self.filenames = []

        for file in files:
            # full = os.path.join(ds_folder, file)
            try:
                # read_image(full)  # to make sure that all image is readable before calling __getitem__
                self.filenames.append(file)

                if max_sample and max_sample == len(self.filenames):
                    break
            except IndexError:
                logger.error(f'failed to extract pid pattern from img {file}')
            except Exception as e:
                logger.exception(e)

        self._split_train_test(train_ratio, split_by_pid)
        del self.filenames

    def read_subsets_from_file(self, all_filenames, path_to_subsets):
        def _get_filename_with_ext(files, prefix):
            for f in files:
                if prefix in f:
                    return f
            return None

        fin = open(os.path.join(PALLET_DS_HOME, path_to_subsets), 'r')
        data = json.load(fin)
        fin.close()

        filter_none = lambda l: [x for x in l if x is not None]
        self.train_set = filter_none([_get_filename_with_ext(all_filenames, f) for f in data["train"]])
        self.test_set = filter_none([_get_filename_with_ext(all_filenames, f) for f in data["test"]])
        self.gallery_set = filter_none([_get_filename_with_ext(all_filenames, f) for f in data["gallery"]])

    def get_subset(self, subset):
        if subset == 'train':
            return self.train_set
        if subset == 'test':
            return self.test_set
        if subset == 'gallery':
            return self.gallery_set
        return None

    def _split_train_test(self, train_ratio: float, by_pid: bool = False):
        """
        split train/test set by pid or img_id
        save result in self variable, `train_set` or `test_set`
        :param train_ratio: a number between (0, 1], indicate ratio of training set to original dataset
            actual size of training set is not calculated by number of samples in original dataset
            but also depend on splitting id (pallet_id or image_id)
        :param by_pid: to split train-test by pallet_id (if True) or image_id (if False),
            2 images with different image_id might share the same pallet_id
        """
        if by_pid:  # split by pallet_id
            img2pid = {file: get_pid(file) for file in self.filenames}
            pids = list(set(img2pid.values()))
            num_pallet_in_train = int(train_ratio * len(pids))
            train_pids = random.choices(pids, k=num_pallet_in_train)

            for idx, img in enumerate(self.filenames):
                pid = img2pid[img]
                if pid in train_pids:
                    self.train_set.append(img)
                else:
                    self.test_set.append(img)
            logger.info(
                f"train-test-split by PID, Train ({num_pallet_in_train} Pallets - {len(self.train_set)} Images), "
                f"Test ({len(pids) - num_pallet_in_train}) - {len(self.test_set)} Images")
        else:  # split by image_id
            ds_size = len(self.filenames)
            train_size = int(ds_size * train_ratio)
            self.train_set, self.test_set = self.filenames[:train_size], self.filenames[train_size:]
            logger.info(f"train-test-split by IID, Train ({len(self.train_set)} Images), "
                        f"Test ({len(self.test_set)} Images")


class PalletDataset(Dataset):
    def __init__(self, ds_folder, cf: DatasetConfig, train: bool = True, inference=False,
                 transform=None, crop=None, subset: str = None, load_extension=False, **kwargs):
        """
        load image dataset from a raw folder and convert to Pytorch's dataset
        :param ds_folder: path to dataset's folder
        :param cf: dataset config which gives information about train/test subset
        :param transform: transform operator
        :param crop: crop original image into smaller parts
        :param load_extension: return image's extension along
        """
        self.home_folder = ds_folder

        if crop:  # todo
            self.crop = crop

        self.transform = transform
        self.load_ext = load_extension

        self.subset = subset
        self.is_train_set = train
        if subset:
            self.is_train_set = True if subset == 'train' else False
        else:
            self.subset = 'train' if train else 'test'
        self.inference = inference

        self.filenames = cf.get_subset(self.subset)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        # TODO: implement cropping operator in-place
        img_path = os.path.join(self.home_folder, self.filenames[idx])
        img = read_image(img_path)

        if self.transform:
            try:
                img = self.transform(img)
            except Exception as e:
                logger.debug('error at {}: {}'.format(img_path, e))
                logger.exception(e)
                return None

        if self.load_ext:
            name, extension = os.path.splitext(os.path.basename(img_path))
            return img, name, extension
        else:
            return img

    def __repr__(self):
        return 'PALLET {} DATASET:\n' \
               '- dataset folder: {}\n' \
               '- number of image: {}\n' \
               '- transforms: {}' \
            .format(self.subset.capitalize(),
                    self.home_folder,
                    len(self.filenames),
                    self.transform)


class SiameseDataset(PalletDataset):
    def __init__(self, ds_folder, cf: DatasetConfig, train: bool = True, dup: int = 1,
                 transform=None, crop=None, subset: str = None, **kwargs):
        """
        reuse initialization from parent class
        """
        super().__init__(ds_folder=ds_folder, cf=cf, train=train, transform=transform,
                         crop=crop, subset=subset, **kwargs)

        self.cache = {}
        if self.is_train_set and not self.inference:
            self.samples = generate_triplet_pairs(
                self.filenames,
                [get_pid(file) for file in self.filenames],
                dup=dup)
        else:
            self.samples = self.filenames

    def __len__(self):
        return len(self.samples)

    def cache_transform(self, img_name):
        img = read_image(os.path.join(self.home_folder, img_name))
        if img_name not in self.cache:
            img = self.transform(img)
            self.cache[img_name] = img
        return self.cache[img_name]

    def __getitem__(self, idx):
        """
        if training set, return triplet (anchor, positive, negative)
        if test set, return image along with its pallet_id
        """
        sample = self.samples[idx]

        if self.is_train_set and not self.inference:
            # imgs = [read_image(os.path.join(self.home_folder, filename)) for filename in sample]
            # if self.transform:
            #     imgs = [self.transform(img) for img in imgs]
            imgs = [self.cache_transform(filename) for filename in sample]
            anchor, positive, negative = imgs
            return anchor, positive, negative
        else:
            # img = read_image(os.path.join(self.home_folder, sample))
            # img = self.transform(img)
            img = self.cache_transform(sample)
            return img, get_pid(sample)

    def __repr__(self):
        return 'PALLET {} DATASET:\n' \
               '- dataset folder: {}\n' \
               '- number of {}: {}\n' \
               '- transforms: {}' \
            .format(self.subset.capitalize(),
                    self.home_folder,
                    "triplet" if self.is_train_set else "image",
                    len(self.samples),
                    self.transform)


class SelectiveSiameseDataset(SiameseDataset):
    def __init__(self, ds_folder, cf: DatasetConfig, **kwargs):
        super().__init__(ds_folder, cf, **kwargs)

    def __getitem__(self, idx):
        if self.is_train_set:
            return super().__getitem__(idx) + (idx,)
        return super().__getitem__(idx)
