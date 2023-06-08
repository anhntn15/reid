## Project overview
```text

├── config 
│   ├── autoencoder
│   ├── colorbased
│   │   ├── 2_fc_blocks
│   │   └── selective_triplets
│   ├── siamese_502
│   ├── siamese_gnn
│   └── sift
├── model
│   ├── datasets
│   ├── transform
│   └── utils
├── notes
│   ├── ae
│   └── siamese
├── tmp
├── training
└── utils
    ├── helper_scripts
    ├── image_processing

```

Short description:
- `config`: store all config file. Each config file defines all component for a training process: dataset, feature engineering step, model architecture, hyper-parameter (epoch, lr, loss function), output folder (store model artifact), ...
- `model`:
  + implementation of [Autoencoder](model/autoencoder.py) and [Siamese](model/siamese.py) model, based on Pytorch library.
  + datasets: wrap pallet dataset (image-based and graph-based version) as instance of Pytorch Dataset.
  + transform: do feature engineering in the fly.
  + [config_helper](model/utils/config_helper.py): backbone class that parses infos from config file to init all training components.
- `notes`: records of some experiments.
- `tmp`: store uncommitted files: logs, model artifact, embedding files, ...
- `training`: where things come together, with infos loaded from config files, define actual training/inference procedure.
  + `autoencoder.py`: train an autoencoder model. With a helper function to visualize constructed images from latent space.
  + `inference_embs.py`: from a trained Siamese model, infer embedding vectors for all datapoint in a dataset. Note that the .npz output of training process stores embedding vectors for only _gallery_ and _test_ set, doesn't include _training_ set.
  + `siamese502.py`: train a Siamese model for image-based version. Randomly split train/test/gallery from origin whole dataset.
  + `siamese502_fold.py`: train a Siamese model for image-based version. Load train/test/gallery set from pre-defined files (json).
  + `siamese_gnn.py`: train a Siamese graph neural network for graph-based version.
- `utils`: helper functions, details see [utils](utils)

## Setup
* prepare the local config file: `cd config && cp config.json.template config.json`
* set path to _pallet dataset folder_ by modifying local file `config.json`.
* prepare virtual env, install all required dependencies: `pip install -r requirements.txt`

All dataset versions (light-on, light-off, graph-based, k-fold, ...) are expected to put in the same parent folder, the local config file stores path to this parent folder. The parent folder of all dataset can be structured as:
```text
.
└── camera-0-light-off
└── camera-1-light-on
└── graphplus
│   └── base
│   └── medium
│   └── off
│   └── s1
│   └── s10
│   └── s15
│   └── smaller
└── graphrepr
└── graphrepr2
└── kfold_fn

```

## Training
All logs will be recorded in `tmp/debug.log`. As multiple processes can modify the same logging file at the same time, information flow can be messed up, there is an option to add suffix to create new logging file, details see [log_config.py](utils/log_config.py#L7).

### Train autoencoder model
Experiment with [AE models](model/autoencoder.py) and image version of pallet dataset:
  * create new config file to define your own parameter, or directly experiment with existing .yml files in [config folder](config/autoencoder).
  * example: `python training/autoencoder.py -cf config/autoencoder/CMR_cam_on.yml`

### Train SiameseGNN model
Experiment with variants of [Siamese models](model/siamese.py).
  * see example at [config file](config/siamese_gnn/siamese_gnn.yml)
  * to train and generate emb files, run `python training/siamese_gnn.py train -cf config/siamese_gnn/siamese_gnn.yml`
  * to evaluate rank-k accuracy, run `python training/siamese_gnn.py eval -cf config/siamese_gnn/siamese_gnn.yml`

### Train Siamese502 crossvalidation
  * to train and generate emb files: `python training/siamese502_fold.py -cf config/siamese_502/siamese502_kfold_0.yml`

## Inference
The required inputs for inference step are:
- the config file that used for training step. From the config file, model architecture, training dataset, folder that contains all trained models are determined. All trained models will be loaded, infer all data point in the original dataset, then save output to the current working folder.
- type of data representation: `gnn` for graph and `502` for images.

Example: `python training/inference_embs.py -t gnn -cf tmp/exp1/gnn_fold_0.yml tmp/exp1/gnn_fold_1.yml`

## Evaluation
Only the evaluation of re-identification tasks are implemented. By default, rank-{1,6,10} accuracy will be calculated.

In practice, model and embedding files are stored in `tmp` directory. It's faster to copy the script `utils/helper_scripts/rank_k_acc.py` to `tmp` and evaluate here. For example:
- `cp utils/helper_scripts/rank_k_acc.py tmp/`
- `cd tmp && python rank_k_acc.py exp1`

