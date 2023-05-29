import argparse
import os

import numpy as np
import torch
from torch.utils.data import Subset, DataLoader

from model.utils.config_helper import Siamese502FoldCR
from training import DEVICE
from utils.log_config import get_custom_logger

logger = get_custom_logger(__name__, 'DEBUG')


def train_siamese(cf_file: str):
    logger.info('NORMAL TRAINING MODE.')
    cf_reader = Siamese502FoldCR(cf_file)
    cf_reader.load_inducer_components()

    train_loader, test_loader, gallery_loader = cf_reader.get_data_loader()
    model = cf_reader.get_model()
    logger.info(f'DEVICE: {DEVICE}')
    model.to(DEVICE)

    criterion, optimizer, epochs, model_path = cf_reader.get_inducer_config()

    for epoch in range(1, epochs + 1):
        total_loss = 0
        model.train()
        for batch in train_loader:
            a, p, n = batch
            anchors, positives, negatives = a.to(DEVICE), p.to(DEVICE), n.to(DEVICE)
            emb_anchors, emb_positives, emb_negatives = model(anchors, positives, negatives)
            loss = criterion(emb_anchors, emb_positives, emb_negatives)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        logger.info('Epoch:{}, Loss:{:.4f}'.format(epoch, total_loss / len(train_loader)))

        if epoch % 5 == 0 or epoch == epochs:
            if not os.path.exists(model_path):
                os.makedirs(model_path)

            tmp_path = f'{model_path}_{epoch}'
            torch.save(model, f'{tmp_path}.pt')
            get_embeddings(model, tmp_path, test_loader, gallery_loader)


def selective_train_siamese(cf_file: str):
    cf_reader = Siamese502FoldCR(cf_file)
    cf_reader.load_inducer_components()

    train_loader, test_loader, gallery_loader = cf_reader.get_data_loader()
    model = cf_reader.get_model()
    logger.info(f'DEVICE: {DEVICE}')
    model.to(DEVICE)

    criterion, optimizer, epochs, model_path = cf_reader.get_inducer_config()

    train_ds = train_loader.dataset
    init_loss = 2
    triplet_loss = {i: init_loss for i in range(0, len(train_ds))}  # {triplet_id: loss}
    num_train_instance = int(0.5 * len(train_ds))  # 10% of whole dataset -> 50%
    pre_selected_ids = []  # overlapped samples between 2 epochs

    logger.info(f'SELECTIVE TRAINING MODE: use whole triplet every 5 iterations, '
                f'otherwise, select only {num_train_instance} triples.')

    for epoch in range(1, epochs + 1):
        # prepare triplet for next training
        selected_ids = [i for i, v in sorted(triplet_loss.items(), key=lambda item: item[1])[-num_train_instance:]]
        sub_ds = Subset(dataset=train_ds, indices=selected_ids)
        sub_train_loader = DataLoader(dataset=sub_ds, batch_size=train_loader.batch_size)
        loss_inc_count = 0

        # start training
        total_loss = 0
        model.train()
        use_whole_ds = (epoch % 5 == 1)
        data_loader = train_loader if use_whole_ds else sub_train_loader
        for batch in data_loader:
            a, p, n, indices = batch
            anchors, positives, negatives = a.to(DEVICE), p.to(DEVICE), n.to(DEVICE)
            emb_anchors, emb_positives, emb_negatives = model(anchors, positives, negatives)
            loss = criterion(emb_anchors, emb_positives, emb_negatives)

            # record triplet loss
            for i, l in zip(indices, loss):
                if triplet_loss[i.item()] < l.item():  loss_inc_count += 1
                triplet_loss[i.item()] = l.item()

            loss = loss.mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        false_measure = len([x for x in triplet_loss.values() if x > 0])
        logger.info('Epoch:{}, Loss:{:.4f} ({} triplets) - Loss_inc_count: {} - false_count: {} - re-sampled: {}'
                    .format(epoch, total_loss / len(data_loader), len(data_loader.dataset), loss_inc_count,
                            false_measure if use_whole_ds else "", len(set(selected_ids) & set(pre_selected_ids))))
        pre_selected_ids = selected_ids

        if epoch % 5 == 0 or epoch == epochs:
            if not os.path.exists(model_path):
                os.makedirs(model_path)

            tmp_path = f'{model_path}_{epoch}'
            torch.save(model, f'{tmp_path}.pt')
            get_embeddings(model, tmp_path, test_loader, gallery_loader)

    # model = torch.load(model_path)


def get_embeddings(model, output, test_loader, gallery_loader):
    """
    get new embedding for gallery/test set,
    save embeddings in .npz file
    """
    model.eval()
    with torch.no_grad():
        test_embs, test_pids = [], []
        for (img, pid) in test_loader:
            img = img.to(DEVICE)
            emb = model.feed(img).cpu().numpy()
            test_embs.append(emb)
            test_pids.append(pid)

        g_embs, g_pids = [], []
        for (img, pid) in gallery_loader:
            img = img.to(DEVICE)
            emb = model.feed(img).cpu().numpy()
            g_embs.append(emb)
            g_pids.append(pid)

    output = f'{output}.npz'
    np.savez_compressed(output,
                        temb=np.asarray(test_embs, dtype=float), tid=test_pids,
                        gemb=np.asarray(g_embs, dtype=float), gid=g_pids)
    logger.info(f'save embedding in {output}')


parser = argparse.ArgumentParser(description='Train Siamese model with Pallet502 dataset from .yml file')
parser.add_argument('-cf', '--config',
                    nargs=1,
                    type=str,
                    help="path to model's config file")
parser.add_argument('-m', '--mode', nargs=1, type=str, choices=['normal', 'selective'],
                    help="mode to train:\t"
                         "`normal` uses whole triplet, `selective` uses 10% of triplets with highest loss.")

arg = parser.parse_args()
if arg.config:
    print('loading config file:', arg.config[0])
    if arg.mode == 'normal':
        train_siamese(arg.config[0])
    elif arg.mode == 'selective':
        selective_train_siamese(arg.config[0])
    else:
        print('accept mode: normal|selective')
else:
    print('training/siamese502_fold.py -h')
