import argparse

import numpy
import torch

from model.utils.config_helper import SiameseGNNConfigReader
from training import DEVICE
from utils.log_config import get_custom_logger

logger = get_custom_logger(__name__, 'DEBUG')


def train_model(cf_file):
    cf_reader = SiameseGNNConfigReader(cf_file)
    cf_reader.load_inducer_components()

    train_loader, test_loader, gallery_loader = cf_reader.get_data_loader()
    model = cf_reader.get_model()
    model.to(DEVICE)

    criterion, optimizer, epochs, model_path = cf_reader.get_inducer_config()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        for (a, p, n) in train_loader:
            anchors, positives, negatives = a.to(DEVICE), p.to(DEVICE), n.to(DEVICE)
            emb_anchors, emb_positives, emb_negatives = model(anchors, positives, negatives)
            loss = criterion(emb_anchors, emb_positives, emb_negatives)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        logger.info('Epoch:{}, Loss:{:.4f}'.format(epoch, total_loss / len(train_loader)))
        if epoch % 5 == 0 or epoch == epochs:
            tmp_path = f'{model_path}_{epoch}'
            torch.save(model, f'{tmp_path}.pt')
            generate_embs(model=model, model_path=tmp_path, test_loader=test_loader, gallery_loader=gallery_loader)

    # torch.save(model, model_path)
    # generate_embs(model=model, model_path=model_path, test_loader=test_loader, gallery_loader=gallery_loader)

    train_loader.dataset.clean_processed_dir()


def generate_embs(cf_file=None, model=None, model_path=None, test_loader=None, gallery_loader=None):
    if cf_file:
        cf_reader = SiameseGNNConfigReader(cf_file)
        cf_reader.load_inducer_components()
        _, _, _, model_path = cf_reader.get_inducer_config()
        model = torch.load(model_path)

        _, test_loader, gallery_loader = cf_reader.get_data_loader()
    model.eval()
    with torch.no_grad():
        test_embs, test_pids = [], []
        for (graph, pid) in test_loader:
            emb = model.feed(graph.x, graph.edge_index, None)
            test_embs.append(emb.squeeze().numpy())
            test_pids.append(pid)

        gal_embs, gal_pids = [], []
        for (graph, pid) in gallery_loader:
            emb = model.feed(graph.x, graph.edge_index, None)
            gal_embs.append(emb.squeeze().numpy())
            gal_pids.append(pid)

        npz_path = f'{model_path}.npz'
        numpy.savez_compressed(npz_path,
                               temb=numpy.asarray(test_embs, dtype=float), tid=test_pids,
                               gemb=numpy.asarray(gal_embs, dtype=float), gid=gal_pids)
        print(f'saved embedding in {npz_path}')


def evaluate_rank_k(cf_file=None, model_path=None):
    if not model_path:
        cf_reader = SiameseGNNConfigReader(cf_file)
        _, _, _, model_path = cf_reader.get_inducer_config()

    emb_file = f'{model_path}.npz'

    data = numpy.load(emb_file)

    t_embs, t_pids, g_embs, g_pids = data['temb'], data['tid'], data['gemb'], data['gid']
    if t_embs.ndim == 3:
        t_embs = t_embs.mean(axis=1)
        g_embs = g_embs.mean(axis=1)

    # for each image in test set, find its closet image in gallery, 400x100 (row - test, col - gallery)
    deltas = []
    dexs = []
    for x, i in zip(t_embs, t_pids):
        dist = (x - g_embs) ** 2
        dist = numpy.mean(dist, axis=-1)
        order = numpy.argsort(dist)
        delta, dex = [[j, dex] for j, dex in enumerate(order) if g_pids[dex] == i][0]
        deltas.append(delta)
        dexs.append(dex)

    def findfrac(deltas, test):
        return numpy.round(numpy.mean([1 if t < test else 0 for t in deltas]), 3)

    ranks = [1, 6, 10]
    acc = [findfrac(deltas, k) for k in ranks]
    logger.info(f"rank-{ranks} acc: {acc}")

    return acc


def train(arg):
    train_model(arg.config[0])


def emb(arg):
    generate_embs(cf_file=arg.config[0])


def eval(arg):
    evaluate_rank_k(cf_file=arg.config[0])


def kfold(arg):
    cross_validation(cf_files=arg.config)


def cross_validation(cf_files: list):
    print(cf_files)
    record = []
    for cf in cf_files:
        train_model(cf)
        acc = evaluate_rank_k(cf)
        record.append(acc)

    logger.info(f'acc for model in {cf_files}:\t{record}')
    logger.info(f'average: {numpy.round(numpy.array(record).mean(axis=0), 3).tolist()}')


parser = argparse.ArgumentParser(description='Train SiameseGNN model with Graph dataset from .yml file', add_help=False)
parser.add_argument('-cf', '--config', nargs='+', type=str, help="path to model's config file")

subparsers = parser.add_subparsers(help='training, generate embedding, run evaluation')

parser_train = subparsers.add_parser("train", parents=[parser])
parser_train.set_defaults(func=train)

parser_emb = subparsers.add_parser("emb", parents=[parser])
parser_emb.set_defaults(func=emb)

parser_eval = subparsers.add_parser("eval", parents=[parser])
parser_eval.set_defaults(func=eval)

parser_eval = subparsers.add_parser("kfold", parents=[parser])
parser_eval.set_defaults(func=kfold)

arg = parser.parse_args()
arg.func(arg)
