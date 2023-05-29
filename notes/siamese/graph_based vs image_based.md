Note on experiments with Siamese models for 2 versions of representation:
- image-based version
- graph-based version

Name format of .npz files: {experiment-type}-{dataset-type}-{fold-index}_{snapshot_epoch}.npz. Example with `siamese502-kfold-on-0_28.npz`:
- siamese502: Siamese model + image-based version
- kfold|fold: experiment on k-fold
- on: light-on dataset version
- 0: first fold
- 28: embedding is retrieved when model is trained at epoch 28

## Siamese502
### light-on kfold
#### ResizeRatioPadding
siamese502-kfold-on-0_28.npz: [0.3069306930693069, 0.6435643564356436, 0.7896039603960396]

siamese502-kfold-on-1_30.npz: [0.3025, 0.7825, 0.8725]

siamese502-kfold-on-2_16.npz: [0.26804123711340205, 0.6958762886597938, 0.7860824742268041]

siamese502-kfold-on-3_8.npz: [0.24744897959183673, 0.6913265306122449, 0.798469387755102]

siamese502-kfold-on-4_2.npz: [0.3225, 0.73, 0.845]

#### Resize (normal)
siamese502-kfold-on-0_20.npz: [0.28465346534653463, 0.6806930693069307, 0.7648514851485149]

siamese502-kfold-on-0_5.npz: [0.3069306930693069, 0.6435643564356436, 0.7574257425742574]

siamese502-kfold-on-1_25.npz: [0.325, 0.6975, 0.805]

siamese502-kfold-on-2_15.npz: [0.29381443298969073, 0.711340206185567, 0.8247422680412371]

siamese502-kfold-on-3_10.npz: [0.3112244897959184, 0.6377551020408163, 0.7551020408163265]

siamese502-kfold-on-4_5.npz: [0.1825, 0.5, 0.63]  -- continue ls92

### light-off kfold
siamese502-kfold-off-0_4.npz: [0.25, 0.650990099009901, 0.8044554455445545]

siamese502-kfold-off-1_18.npz: [0.24504950495049505, 0.6039603960396039, 0.7103960396039604]

siamese502-kfold-off-1_20.npz: [0.25495049504950495, 0.5693069306930693, 0.7227722772277227]


## SiameseGNN
### GCN - margin0.2
siamese-gnn-fold-0_35.npz: [0.16, 0.42, 0.54]

siamese-gnn-fold-1_25.npz: [0.25, 0.6075, 0.715]

siamese-gnn-fold-2_32.npz: [0.235, 0.6625, 0.7625]

siamese-gnn-fold-3_15.npz: [0.285, 0.725, 0.8275]

siamese-gnn-fold-3_39.npz: [0.29, 0.665, 0.7525]

siamese-gnn-fold-3_49.npz: [0.31, 0.69, 0.7775]

### GCN - margin0.3
siamese-gnn-fold-0_22.npz: [0.1625, 0.3775, 0.505]

siamese-gnn-fold-1_21.npz: [0.2325, 0.6025, 0.7275]

siamese-gnn-fold-2_16.npz: [0.28, 0.66, 0.7675]

siamese-gnn-fold-3_25.npz: [0.2975, 0.7225, 0.845]


### GAT
siamese-gnn-fold-0_90.npz: [0.255, 0.5, 0.615]

siamese-gnn-fold-1_85.npz: [0.3275, 0.69, 0.7675]

siamese-gnn-fold-1_90.npz: [0.33, 0.6825, 0.765]

siamese-gnn-fold-2_75.npz: [0.295, 0.6225, 0.7375]

siamese-gnn-fold-3_85.npz: [0.3725, 0.725, 0.805]

