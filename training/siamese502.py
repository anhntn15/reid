import argparse

import torch
import numpy as np

from model.utils.config_helper import ConfigReader
from utils.log_config import get_custom_logger

logger = get_custom_logger(__name__, 'DEBUG')


def get_gallery_test_idx(test_loader):
    """
    randomly set instance for gallery/test set
    :return: test indices, gallery indices
    """
    pids = np.array([int(pid[0]) for (_, pid) in test_loader])
    test_idx = np.ones((len(pids),), dtype=bool)
    for lid in set(pids):
        img_idx = np.where(pids == lid)[0]  # get index of imgs have pallet_id `lid`
        g_idx = np.random.choice(img_idx, 1)[0]  # random one image
        test_idx[g_idx] = False  # remove it from test set
    return test_idx, ~test_idx


def train_siamese(cf_file: str):
    cf_reader = ConfigReader(cf_file)
    cf_reader.load_inducer_components()

    train_loader, test_loader = cf_reader.get_data_loader()
    model = cf_reader.get_model()

    criterion, optimizer, epochs, model_path = cf_reader.get_inducer_config()

    test_idx, gallery_idx = get_gallery_test_idx(test_loader)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        for batch in train_loader:
            anchors, positives, negatives = batch
            emb_anchors, emb_positives, emb_negatives = model(anchors, positives, negatives)
            loss = criterion(emb_anchors, emb_positives, emb_negatives)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        logger.info('Epoch:{}, Loss:{:.4f}'.format(epoch, total_loss / len(train_loader)))

        if epoch%5 == 0 or epoch == epochs:
            tmp_path = f'{model_path}_{epoch}'
            torch.save(model, tmp_path)
            get_embeddings(model, test_loader, gallery_idx, test_idx, tmp_path)

    # model = torch.load(model_path)


def get_embeddings(model, test_loader, gallery_idx, test_idx, output):
    """
    get new embedding for gallery/test set,
    save embeddings in .npz file
    """
    model.eval()
    with torch.no_grad():
        embs, pids = [], []
        for (img, pid) in test_loader:
            emb = model.feed(img).numpy()
            embs.append(emb)
            pids.append(pid)

    embs = np.asarray(embs, dtype=float)
    pids = np.asarray(pids, dtype=float)

    output = f'{output}.npz'
    np.savez_compressed(output,
                        temb=embs[test_idx], tid=pids[test_idx],
                        gemb=embs[gallery_idx], gid=pids[gallery_idx])
    logger.info(f'save embedding in {output}')


parser = argparse.ArgumentParser(description='Train Siamese model with Pallet502 dataset from .yml file')
parser.add_argument('-cf', '--config',
                    nargs=1,
                    type=str,
                    help="path to model's config file")

arg = parser.parse_args()
if arg.config:
    print('loading config file:', arg.config[0])
    train_siamese(arg.config[0])
else:
    print('training/siamese502.py -h')
