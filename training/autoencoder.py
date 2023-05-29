import argparse
import os

import torch
from torch.utils.data import DataLoader, RandomSampler

from model.utils.config_helper import ConfigReader
from model.utils.visualization import visualize_constructed_image
from utils.log_config import get_custom_logger

logger = get_custom_logger(__name__, 'DEBUG')


def train_ae_model(config_file):
    """
    train an autoencoder model with config loading from a config file
    :return:
    """
    cf_reader = ConfigReader(config_file)
    cf_reader.load_inducer_components()

    train_loader, test_loader = cf_reader.get_data_loader()
    model = cf_reader.get_model()

    criterion, optimizer, epochs, model_path = cf_reader.get_inducer_config()

    for epoch in range(1, epochs + 1):
        total_loss = 0
        for batch in train_loader:
            reconstructed_imgs = model(batch)
            loss = criterion(reconstructed_imgs, batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        logger.info('Epoch:{}, Loss:{:.4f}'.format(epoch, total_loss / len(train_loader)))

    torch.save(model, model_path)

    if test_loader:
        model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in test_loader:
                reconstructed_imgs = model(batch)
                loss = criterion(reconstructed_imgs, batch)
                total_loss += loss.item()
        logger.info('Loss on Test-set: {:.4f}'.format(total_loss / len(test_loader)))

    # sample 200 images from test-set to visualize
    ds = test_loader.dataset if test_loader else train_loader.dataset
    ds.load_ext = True
    randomer = RandomSampler(ds, num_samples=200)
    sample_loader = DataLoader(ds, sampler=randomer, batch_size=1)
    model_name, _ = os.path.splitext(os.path.basename(model_path))
    visualize_constructed_image(model, sample_loader, torch.nn.MSELoss(), model_name + '_visualize')


def reconstruct_from_other_latent_space(cf_dataset, cf_model, out, num: int = 100):
    """
    calculate L(x1, g2(f2(x1))) or L(x2, g1(f1(x2)))
    used the trained model to reconstruct image from another dataset
    saving the reconstructed images and loss distribution to visualize later
    """
    inp_cf = ConfigReader(cf_dataset)
    out_cf = ConfigReader(cf_model)

    ds = inp_cf.get_data_loader()[0].dataset
    ds.load_ext = True  # force _get_item_ return correct image.extension
    randomer = RandomSampler(ds, num_samples=num)
    data_loader = DataLoader(ds, sampler=randomer, batch_size=1)

    _, _, _, model_path = out_cf.get_inducer_config()
    model = torch.load(model_path)
    visualize_constructed_image(model, data_loader, torch.nn.MSELoss(), out)


parser = argparse.ArgumentParser(description='Train AE model from .yml file')
parser.add_argument('-cf', '--configs',
                    nargs='+', type=str,
                    help='list of model config files to train')

args = parser.parse_args()
if args.configs:
    print(args.configs)
    for cf in args.configs:
        train_ae_model(cf)
else:
    print('training/autoencoder.py -h')
