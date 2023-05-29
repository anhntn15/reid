"""
split origin dataset into k disjoint subset (fold)
different folds are then used in cross-validation.
"""
import numpy as np


def get_full_data(file_path):
    g = np.load(file_path)
    fxA = np.append(g['xA'], g['gxA'], axis=0)
    fxA = np.append(fxA, g['txA'], axis=0)
    fxX = np.append(g['xX'], g['gxX'], axis=0)
    fxX = np.append(fxX, g['txX'], axis=0)
    fi = np.append(g['i'], g['gi'], axis=0)
    fi = np.append(fi, g['ti'], axis=0)

    print(f'full data:\nfxA: {len(fxA)}\nfxX: {len(fxX)}\nfi: {len(fi)}')
    return fxA, fxX, fi


def random_k_fold(xA, xX, pid, k=4, out_folder: str = '.'):
    """
    split full dataset into 4 folds
    - 3 folds ~ 60%: training
    - 1 fold ~ 40%: gallery + test
    """
    uniq_pids = list(set(pid.tolist()))
    to_chunk = lambda l, n: [l[i:i + n] for i in range(0, len(l), n)]
    chunk_size = int(len(uniq_pids) / k)
    chunks = to_chunk(uniq_pids, chunk_size)

    print(f'CHUNKS: {[len(c) for c in chunks]}')

    for ci in range(0, len(chunks)):
        if len(chunks[ci]) < chunk_size:
            continue
        train_pids = []
        for oi in range(0, len(chunks)):
            if oi != ci:
                train_pids.extend(chunks[oi])
        leaveout_pids = np.asarray(chunks[ci])

        train_idx = np.isin(pid, train_pids)
        leaveout_idx = ~train_idx
        gallery_idx = []
        for lid in leaveout_pids:
            img_idx = np.where(pid == lid)[0]  # get index of imgs have pallet_id `lid`
            g_idx = np.random.choice(img_idx, 1)[0]  # random one image
            # print('img_idx', img_idx, '\t', g_idx)
            gallery_idx.append(g_idx)  # put it into gallery
            leaveout_idx[g_idx] = False  # remove it from leaveout set

        train_xA, train_xX, train_i = xA[train_idx], xX[train_idx], pid[train_idx]
        test_xA, test_xX, test_i = xA[leaveout_idx], xX[leaveout_idx], pid[leaveout_idx]
        gallery_xA, gallery_xX, gallery_i = xA[gallery_idx], xX[gallery_idx], pid[gallery_idx]

        assert set(train_i.tolist()).isdisjoint(test_i.tolist()), 'train and test must be disjoint pallet_id'

        print(f'FOLD {ci}:\n'
              f'\ttrain: {train_xA.shape}, {train_xX.shape}, {train_i.shape}\n'
              f'\ttest: {test_xA.shape}, {test_xX.shape}, {test_i.shape}\n'
              f'\tgallery: {gallery_xA.shape}, {gallery_xX.shape}, {gallery_i.shape}\n')

        out_npz_file = f'{out_folder}/fold_{ci}.npz'
        np.savez_compressed(out_npz_file,
                            xA=train_xA, xX=train_xX, i=train_i,
                            gxA=gallery_xA, gxX=gallery_xX, gi=gallery_i,
                            txA=test_xA, txX=test_xX, ti=test_i)


fxA, fxX, fi = get_full_data('../../../pallet-block-502/graphrepr/anhgraph.npz')
random_k_fold(xA=fxA, xX=fxX, pid=fi, k=4, out_folder='../../pallet-block-502/graphrepr/')
