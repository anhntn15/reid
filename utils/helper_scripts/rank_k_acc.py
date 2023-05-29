"""
calculate rank-{1,6,10} accuracy of re-identification task
target input is path of folder that contains at least one embedding files (.npz)
the .npz file must contain embedding vector of `gallery` and `test` set.
"""
import os
import sys

import numpy as np

# folder to process can be passed as argument
# e.g: `python rank_k_acc.py emb_folder_name`
folder = sys.argv[1] if len(sys.argv) > 0 else '.'

for f in os.listdir(folder):
    if not f.endswith('.npz'): continue
    emb_file = f'{folder}/{f}'

    data = np.load(emb_file)

    t_embs, t_pids, g_embs, g_pids = data['temb'], data['tid'], data['gemb'], data['gid']
    if t_embs.ndim == 3:
        t_embs = t_embs.mean(axis=1)
        g_embs = g_embs.mean(axis=1)
    # for each image in test set, find its closet image in gallery, 400x100 (row - test, col - gallery)
    deltas = []
    dexs = []
    for x, i in zip(t_embs, t_pids):
        dist = (x - g_embs) ** 2
        dist = np.mean(dist, axis=-1)
        order = np.argsort(dist)
        delta, dex = [[j, dex] for j, dex in enumerate(order) if g_pids[dex] == i][0]
        deltas.append(delta)
        dexs.append(dex)


    def findfrac(deltas, test):
        return np.mean([1 if t < test else 0 for t in deltas])


    rank_acc = [findfrac(deltas, k) for k in [1, 6, 10]]
    print(f'{f}: {rank_acc}')
