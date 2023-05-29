"""
combine representation vector from 2 embedding files (.npz)
(as ensemble step with 2 choices: concatenate or average)
"""
import sys

import numpy as np


def combine(inp1: str, inp2: str, out: str, mode='concat'):
    """

    :param inp1: path to first emb file
    :param inp2: path to second emb file
    :param out: path to save result
    :param mode:
     - concat: concat emb vector of second file to the first vector.
     - avg: average emb vectors of 2 files
    :return:
    """
    print(f'{mode} {inp1} vs {inp2} => {out}')

    def process(v1, v2):
        if mode == 'concat':
            return np.expand_dims(normalize(np.concatenate((v1.flatten(), v2.flatten()))), axis=0)
        if mode == 'avg':
            return normalize(np.mean(np.array([v1, v2]), axis=0))
        return None

    g1 = np.load(inp1, allow_pickle=True)
    g2 = np.load(inp2, allow_pickle=True)
    emb_result = {}

    for prefix in ['', 'g', 't']:
        emb1 = {}
        for fn, emb in zip(g1[f'{prefix}fn'], g1[f'{prefix}emb']):
            emb1[fn] = emb
        res = []
        for fn, emb in zip(g2[f'{prefix}fn'], g2[f'{prefix}emb']):
            v = process(emb1[fn], emb)
            res.append(v)

        emb_result[prefix] = np.asarray(res)
    np.savez_compressed(out,
                        emb=np.asarray(emb_result[''], dtype=float), id=g2['emb'], fn=g2['fn'],
                        temb=np.asarray(emb_result['t'], dtype=float), tid=g2['tid'], tfn=g2['tfn'],
                        gemb=np.asarray(emb_result['g'], dtype=float), gid=g2['gid'], gfn=g2['gfn'])


def normalize(v):
    """
    normalize vector to mean 0, std 1
    """
    mean, std = np.mean(v), np.std(v)
    return (v - mean) / std


# usage: file1, file2, output_file, mode (concat|avg)
# e.g: `python ensemble.py inp1 inp2 output concat`
combine(str(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4])
