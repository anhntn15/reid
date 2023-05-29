"""
graph dataset notation:
- xA: training - adj matrix (1505x50x50)
- xX: training - graph feature (1505x50x10)
- i: pallet id in training data
- gxA: gallery - adj matrix (100x50x50)
- gxX: gallery - graph feature (100x50x10)
- gi: pallet id in gallery data
- txA: testing - adj matrix (400x50x50)
- txX: testing - graph feature (400x50x10)
- ti: pallet id in testing data
"""
import enum
import os.path
import shutil
from typing import Union, List, Tuple

import numpy as np
import torch

from torch_geometric.data import Data, Dataset

from model.datasets import generate_triplet_pairs


class SubsetType(enum.Enum):
    TRAIN = 'train'
    TEST = 'test'
    GALLERY = 'gallery'


class SiameseGNNDataset(Dataset):
    def __init__(self, root: str, subset: SubsetType, raw_file: str, dup: int = 1, inference=False):
        """
        read dataset from .npz file
        :param root: folder contains .npz file
        :param subset: train/test/gallery
        :param raw_file: name of data file contains graph dataset
        :param dup: number of picking negative sample for same (anchor, positive) pair when generating triplets
        """
        self.subset_type = subset if isinstance(subset, SubsetType) else SubsetType(subset)
        self.raw_file = raw_file
        self.data_files = ['not_found.pt']
        self.dup = dup
        self.inference = inference
        super(SiameseGNNDataset, self).__init__(root=root)

    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple]:
        return [self.raw_file]

    @property
    def processed_file_names(self):
        return self.data_files

    def get_graph_instances(self):
        """
        get relevant files for each subset
        :return: feature graph, adj matrix, pallet id
        """
        f = np.load(os.path.join(self.root, self.raw_file))

        if self.subset_type == SubsetType.TRAIN:
            return f['xX'], f['xA'], f['i']

        if self.subset_type == SubsetType.TEST:
            return f['txX'], f['txA'], f['ti']

        if self.subset_type == SubsetType.GALLERY:
            return f['gxX'], f['gxA'], f['gi']

    def process(self):
        graph_list = []
        features, adjs, ids = self.get_graph_instances()
        self.filenames = ids
        num_graph = ids.shape[0]

        # create torch.Data instance for each graph
        for i in range(num_graph):
            row, col = np.nonzero(adjs[i])
            row = torch.from_numpy(row).to(torch.long)
            col = torch.from_numpy(col).to(torch.long)
            edge_index = torch.stack([row, col], dim=0)
            x = torch.tensor(features[i], dtype=torch.float)
            data = Data(x=x, edge_index=edge_index)
            graph_list.append(data)

        self.data_files = []
        # create graph-triplets

        if self.subset_type == SubsetType.TRAIN and not self.inference:
            indices = list(range(num_graph))
            triplets = generate_triplet_pairs(indices, ids, self.dup)
            print(f'{len(triplets)} triplets created')
            for i in range(len(triplets)):
                a, p, n = triplets[i]
                triplet = [graph_list[a], graph_list[p], graph_list[n]]
                triplets[i] = triplet

                torch.save(triplet, os.path.join(self.processed_dir, f'data_{i}.pt'))
                self.data_files.append(f'data_{i}.pt')
        else:
            for i, graph in enumerate(graph_list):
                torch.save(graph, os.path.join(self.processed_dir, f'{self.subset_type.value}_{ids[i]}_{i}.pt'))
                self.data_files.append(f'{self.subset_type.value}_{ids[i]}_{i}.pt')

    def clean_processed_dir(self):
        """
        delete all files in `self.processed_dir` since they are temporary files
        """
        shutil.rmtree(self.processed_dir)
        print(f'remove tmp dir: {self.processed_dir}')

    def len(self):
        return len(self.data_files)

    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir, self.data_files[idx]))
        if self.subset_type == SubsetType.TRAIN and not self.inference:
            return data

        # return pid for test/gallery set
        pid = self.data_files[idx].split('_')[1]
        return data, pid

    def __repr__(self):
        return 'GRAPH {} DATASET:\n' \
               '- raw data file: {}\n' \
               '- number of {}: {}\n' \
               '- processed folder: {}' \
            .format(self.subset_type.name,
                    self.raw_file,
                    "triplet" if self.subset_type == SubsetType.TRAIN else "graph",
                    len(self.data_files),
                    self.processed_dir)
