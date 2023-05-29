import os

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import ToPILImage

from config import read_config, PALLET_DS_HOME
from model.transform.color_based_feature import ColorBasedFeature
from model.transform.colorbased_sift import ColorbasedSIFT
from model.transform.resize import ResizeRatio, ResizeRatioPad
from model.transform.sift_transform import SIFTTransform
from model.utils import class_importing
from model.datasets.pallet502 import DatasetConfig
from utils.log_config import get_custom_logger

logger = get_custom_logger(__name__, 'DEBUG')


class ConfigReader:
    def __init__(self, inducer_config):
        self._config = read_config(inducer_config)
        self._data_loader = None
        self._model = None
        self._args = {
            'criterion': None,
            'optimizer': None,
            'epoch': None,
            'model_path': None
        }

    def _load_data_loader(self):
        """
        load dataset from image folder, init DataLoader instance for (train_loader, test_loader)
        if train-test-split is not present, train_loader refers to whole dataset, test_loader is null
        """
        logger.info(f"### {self._config['name']}")

        trans_operators = [transforms.ToTensor()]

        if self._config['dataset'].get('grayscale'):
            trans_operators.append(transforms.Grayscale(num_output_channels=1))

        tmp_variables = {}
        if self._config['dataset'].get('transform'):
            for operator in self._config['dataset'].get('transform'):
                if operator['name'] == 'resize':
                    trans_operators.append(transforms.Resize(size=(operator['height'], operator['width'])))
                    tmp_variables['image_size'] = (operator['height'], operator['width'])
                if operator['name'] == 'resize_ratio_pad':
                    trans_operators.append(ResizeRatioPad(**operator))
                if operator['name'] == 'resize_ratio':
                    trans_operators.append(ResizeRatio(ratio=(operator['ratio'])))
                if operator['name'] == 'crop':
                    trans_operators.append(transforms.CenterCrop(size=(operator['height'], operator['width'])))
                    # pallet_ds_args['crop'] = (operator['height'], operator['width'])
                if operator['name'] == 'sift':
                    trans_operators.append(ToPILImage())
                    trans_operators.append(
                        SIFTTransform(nfeatures=operator['nfeatures'], contrastThreshold=operator['contrastThreshold']))
                if operator['name'] == 'colorbased':
                    tf = ColorBasedFeature(h=operator['h'], w=operator['w'], feature_type=operator['feature_type'])
                    num_features = tf.count_num_subimages(img_h=tmp_variables['image_size'][0],
                                                          img_w=tmp_variables['image_size'][1])
                    model_args = self._config['model'].get('args', {})
                    model_args['in_features'] = num_features
                    self._config['model']['args'] = model_args
                    trans_operators.append(tf)
                if operator['name'] == 'colorbased2':
                    color1 = ColorBasedFeature(h=operator['h'], w=operator['w'], feature_type=operator['f1'])
                    color2 = ColorBasedFeature(h=operator['h'], w=operator['w'], feature_type=operator['f2'])

                    tf = ColorbasedSIFT(color1, color2)
                    model_args = self._config['model'].get('args', {})
                    model_args['color_features_1'] = color1.count_num_subimages(img_h=tmp_variables['image_size'][0],
                                                                                img_w=tmp_variables['image_size'][1])
                    model_args['color_features_2'] = color2.count_num_subimages(img_h=tmp_variables['image_size'][0],
                                                                                img_w=tmp_variables['image_size'][1])
                    self._config['model']['args'] = model_args
                    trans_operators.append(tf)

        del tmp_variables

        # trans_operators.append(transforms.Normalize((0.5,), (0.5,)))  # todo: why this operator helps convergence
        ds_args: dict = self._config['dataset'].get('args', {})
        ds_args.update({'ds_folder': os.path.join(PALLET_DS_HOME, self._config['dataset']['folder']),
                        'transform': transforms.Compose(trans_operators)})

        dataset_class = class_importing(self._config['dataset']['class'])
        ds_cf = DatasetConfig(**ds_args)
        ds_args['cf'] = ds_cf

        self._init_data_loader(dataset_class, ds_args)

    def _init_data_loader(self, dataset_class, ds_args):
        ds_train = dataset_class(train=True, **ds_args)
        logger.info(f'loaded Dataset:\n{ds_train}')
        ds_test = dataset_class(train=False, **ds_args)
        logger.info(f'loaded Dataset:\n{ds_test}')

        data_loader_args = {
            'batch_size': self._config['training']['data_loader'].get('batch_size', 32),  # default is 32 if not present
            'shuffle': self._config['training']['data_loader'].get('shuffle', False)
        }

        train_loader = DataLoader(ds_train, **data_loader_args)
        test_loader = DataLoader(ds_test, batch_size=1) if len(ds_test) else None
        self._data_loader = (train_loader, test_loader)

    def _load_model(self):
        my_model = class_importing(self._config["model"]["name"])
        model_params = self._config['model'].get('args', {})
        self._model = my_model(**model_params)

    def _init_inducer_params(self):
        loss_func_data = self._config['training'].get('criterion')

        if isinstance(loss_func_data, dict):
            my_loss_func = class_importing(loss_func_data['name'])
            del loss_func_data['name']
            criterion = my_loss_func(**loss_func_data)  # the rest are parameters
        elif isinstance(loss_func_data, str):
            my_loss_func = class_importing(loss_func_data)
            criterion = my_loss_func()

        self._args['criterion'] = criterion

        self._args['optimizer'] = torch.optim.Adam(self.get_model().parameters(),
                                                   lr=1e-3,
                                                   weight_decay=1e-5)

        self._args['epoch'] = self._config['training'].get('epoch', 1)  # default is 1 epoch
        self._args['model_path'] = os.path.join(self._config['model']['folder'], self._config['name'])

    def load_inducer_components(self):
        """
        read configuration from `inducer_config` and init corresponding component
        """
        if not self._data_loader:
            self._load_data_loader()

        if not self._model:
            self._load_model()
        logger.info(f'loaded MODEL:\n{self._model}')

        self._init_inducer_params()

    def get_data_loader(self):
        if not self._data_loader:
            self._load_data_loader()
        return self._data_loader

    def get_model(self):
        if not self._model:
            self._load_model()
        return self._model

    def get_inducer_config(self):
        """
        :return: loss function, optimizer, #training_epoch, path to save model
        """
        if None in self._args.values():  # if any component is None
            self._init_inducer_params()
        return self._args['criterion'], self._args['optimizer'], self._args['epoch'], self._args['model_path']


class Siamese502FoldCR(ConfigReader):
    def __init__(self, inducer_config):
        super().__init__(inducer_config)

    def _init_data_loader(self, dataset_class, ds_args):
        ds_train = dataset_class(subset='train', **ds_args)
        logger.info(f'loaded Dataset:\n{ds_train}')
        ds_test = dataset_class(subset='test', **ds_args)
        logger.info(f'loaded Dataset:\n{ds_test}')
        ds_gallery = dataset_class(subset='gallery', **ds_args)
        logger.info(f'loaded Dataset:\n{ds_gallery}')

        data_loader_args = {
            'batch_size': self._config['training']['data_loader'].get('batch_size', 32),  # default is 32 if not present
            'shuffle': self._config['training']['data_loader'].get('shuffle', False)
        }

        train_loader = DataLoader(ds_train, **data_loader_args)
        test_loader = DataLoader(ds_test, batch_size=1)
        gallery_loader = DataLoader(ds_gallery, batch_size=1)
        self._data_loader = (train_loader, test_loader, gallery_loader)


class SiameseGNNConfigReader(ConfigReader):
    def __init__(self, inducer_config):
        super(SiameseGNNConfigReader, self).__init__(inducer_config)

    def _load_data_loader(self):
        logger.info(f"### {self._config['name']}")

        ds_args = self._config['dataset'].get('args', {})
        ds_args['root'] = os.path.join(PALLET_DS_HOME, self._config['dataset']['folder'])

        dataset_class = class_importing(self._config['dataset']['class'])

        ds_train = dataset_class(subset='train', **ds_args)
        logger.info(f'loaded Dataset:\n{ds_train}')
        ds_test = dataset_class(subset='test', **ds_args)
        logger.info(f'loaded Dataset:\n{ds_test}')
        ds_gallery = dataset_class(subset='gallery', **ds_args)
        logger.info(f'loaded Dataset:\n{ds_gallery}')

        data_loader_args = {
            'batch_size': self._config['training']['data_loader'].get('batch_size', 32),  # default is 32 if not present
            'shuffle': self._config['training']['data_loader'].get('shuffle', False)
        }

        import torch_geometric
        train_loader = torch_geometric.loader.DataLoader(ds_train, **data_loader_args)
        test_loader = torch_geometric.loader.DataLoader(ds_test, batch_size=1, shuffle=False)
        gallery_loader = torch_geometric.loader.DataLoader(ds_gallery, batch_size=1, shuffle=False)
        self._data_loader = (train_loader, test_loader, gallery_loader)
