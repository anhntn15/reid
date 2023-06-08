"""
graph dataset notation:
- x: training - feature matrix
- fn: training - file names
- i: id in training data
- tx: test - feature matrix
- tn: test - file names
- ti: id in test data
- gx: gallery - feature matrix
- gn: gallery - file names
- gi: id in gallery data
"""
import enum
import os.path

import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms

from model.datasets import generate_triplet_pairs


class SubsetType(enum.Enum):
    TRAIN = 'train'
    TEST = 'test'
    GALLERY = 'gallery'


class SiameseMetalDataset(Dataset):
    def __init__(self, ds_folder: str, subset: SubsetType, raw_file: str, dup: int = 1, inference=False, **kwargs):
        """
        read dataset from .npz file
        :param root: folder contains .npz file
        :param subset: train/test/gallery
        :param raw_file: name of data file contains all dataset
        :param dup: number of picking negative sample for same (anchor, positive) pair when generating triplets
        """
        super(SiameseMetalDataset, self).__init__()
        self.folder = ds_folder
        self.subset_type = subset if isinstance(subset, SubsetType) else SubsetType(subset)
        self.raw_file = raw_file
        self.triplets = []
        self.dup = dup
        self.inference = inference

        self.process()

    def read_features(self):
        """
        get relevant files for each subset
        :return: feature graph, adj matrix, pallet id
        """
        f = np.load(os.path.join(self.folder, self.raw_file))

        if self.subset_type == SubsetType.TRAIN:
            return f['x'], f['i']

        if self.subset_type == SubsetType.TEST:
            return f['tx'], f['ti']

        if self.subset_type == SubsetType.GALLERY:
            return f['gx'], f['gi']

    def process(self):
        t = transforms.ToTensor()
        features, ids = self.read_features()
        features = [t(i) for i in features]

        # create graph-triplets
        if self.subset_type == SubsetType.TRAIN and not self.inference:
            indices = list(range(ids.shape[0]))
            triplets = generate_triplet_pairs(indices, ids, self.dup)
            print(f'{len(triplets)} triplets created')
            for i in range(len(triplets)):
                a, p, n = triplets[i]
                triplet = [features[a], features[p], features[n]]
                self.triplets.append(triplet)
        else:
            for image, id in zip(features, ids):
                self.triplets.append((image, id))

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, item):
        return self.triplets[item]

    def __repr__(self):
        return 'META {} DATASET:\n' \
               '- raw data file: {}\n' \
               '- number of {}: {}\n' \
            .format(self.subset_type.name,
                    self.raw_file,
                    "triplets" if self.subset_type == SubsetType.TRAIN else "images",
                    len(self.triplets))
