import argparse
import json
import os.path
from typing import List

import numpy as np

import torch

from model.utils.config_helper import SiameseGNNConfigReader, Siamese502FoldCR


def search(folder: str, prefix: str, ext: str = None) -> List[str]:
    """
    search in folder for file with prefix, and/or extension
    :return: list of file names with fullpath
    """
    prefix = prefix.lower()
    ext = ext.lower() if ext else None

    def valid(filename: str):
        if not filename.lower().startswith(prefix): return False
        if ext and (not filename.lower().endswith(ext)): return False

        return True
    res = []
    for f in os.listdir(folder):
        path = os.path.join(folder, f)
        if os.path.isfile(path) and valid(f):
            res.append(path)
    return res


def inference(cf: str, type: str):
    """
    read a folder
    :return:
    """
    if type == '502':
        cf_reader = Siamese502FoldCR(cf)
    else:
        cf_reader = SiameseGNNConfigReader(cf)
    # config for inference mode
    cf_reader._config['dataset']['args']['inference'] = True
    cf_reader._config['dataset']['args']['dup'] = 1
    cf_reader._config['training']['data_loader']['shuffle'] = False
    cf_reader._config['training']['data_loader']['batch_size'] = 1

    cf_reader.load_inducer_components()

    train_loader, test_loader, gallery_loader = cf_reader.get_data_loader()
    # train_tuples = [(file, get_pid(file)) for file in train_loader.dataset.filenames]
    _, _, _, model_path = cf_reader.get_inducer_config()
    folder, name = os.path.split(model_path)

    model_paths = search(folder, prefix=name, ext='.pt')
    print('read models:', model_paths)

    for model_path in model_paths:
        model = torch.load(model_path, map_location=torch.device('cpu'))
        get_embs(model, model_path, test_loader, gallery_loader, train_loader, type)


def get_embs(model, model_path, test_loader, gallery_loader, train_loader, type: str):
    """
    get new embedding for train/gallery/test set,
    save embeddings in .npz file
    """
    model.eval()
    with torch.no_grad():
        train_embs, train_pids = [], []
        for (batch, pid) in train_loader:
            if type == '502':
                emb = model.feed(batch).cpu().numpy()
            if type == 'gnn':
                emb = model.feed(batch.x, batch.edge_index, None)
                emb = emb.squeeze().numpy()

            train_embs.append(emb)
            train_pids.append(pid)

        test_embs, test_pids = [], []
        for (batch, pid) in test_loader:
            if type == '502':
                emb = model.feed(batch).cpu().numpy()
            if type == 'gnn':
                emb = model.feed(batch.x, batch.edge_index, None)
                emb = emb.squeeze().numpy()

            test_embs.append(emb)
            test_pids.append(pid)

        g_embs, g_pids = [], []
        for (batch, pid) in gallery_loader:
            if type == '502':
                emb = model.feed(batch).cpu().numpy()
            if type == 'gnn':
                emb = model.feed(batch.x, batch.edge_index, None)
                emb = emb.squeeze().numpy()
            g_embs.append(emb)
            g_pids.append(pid)

    prefix, _ = os.path.splitext(model_path)
    output = f'{prefix}_all.npz'
    readme = {
        'emb': 'emb for train set',
        'id': 'pallet id for train set',
        'fn': 'filename for train set',
        'temb': 'emb for test set',
        'tid': 'pallet id for test set',
        'tfn': 'filename for test set',
        'gemb': 'emb for gallery set',
        'gid': 'pallet id for gallery set',
        'gfn': 'filename for gallery set',
    }
    with open(f'{prefix}_readme.txt', 'w') as f:
        json.dump(readme, f, indent=4)

    np.savez_compressed(output,
                        emb=np.asarray(train_embs, dtype=float), id=train_pids, fn=train_loader.dataset.filenames,
                        temb=np.asarray(test_embs, dtype=float), tid=test_pids, tfn=test_loader.dataset.filenames,
                        gemb=np.asarray(g_embs, dtype=float), gid=g_pids, gfn=gallery_loader.dataset.filenames)
    print(f'save embedding in {output}')


parser = argparse.ArgumentParser(description='inference: get emb vector for dataset loaded from config file')
parser.add_argument('-cf', '--config',
                    nargs='+',
                    type=str,
                    help="path to model's config file, 1+ file")
parser.add_argument('-t', '--type',
                    type=str,
                    choices=['gnn', '502'],
                    default='gnn',
                    help="type of dataset")

arg = parser.parse_args()
if arg.config:
    for cf in arg.config:
        print('loading config file:', cf)
        inference(cf, type=arg.type)
else:
    print('training/siamese502_fold.py -h')
