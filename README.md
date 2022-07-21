
# Machine Learning in Medical Image Processing Segmentation Implementation

## Introduction

This implementation is based on the 3D-UNet implementation by Adrian Wolny which you can find [here](https://github.com/wolny/pytorch-3dunet). We added an attention layer, a custom dataloader, custom metrics, a new prediction reconstruction, multiple preprocessing steps and multi size input preprocessing.

## Installation

TODOO: Pip freezen und die requirements installieren

## Models

We developed two models. The models have the same underlying architecture, the main difference is that the first model uses the raw full image as input for the train patches, while the second model uses train patches where the aneurysm sizes are aligned.

## Scripts

To generate the training data for the first model run the following script:

```bash
   python ./scripts/generate_train_data.py PATH_TO_TRAIN_DATA
```

Afterwards you can also generate the train data for the second model

```bash
   python ./scripts/generate_train_data_aligned.py PATH_TO_TRAIN_DATA
```

The scripts save the train data as h5 files which have the advantage that you can access
slices of the array without loading the whole array into memory. The downside is, that the file size is much larger so prepare disk space. 10GB should be enough for the train data and val data.

With the train data prepared we can start the training.

To start the training call

```bash
    python start.py --config PATH_TO_CONFIG
```

You can find the config files for the two models in the train-configs folder.

TODOOOOOO
For the first model use the _______ config and for the second model the multisize_config.yaml config.
You need to adjust the file path in loaders/train/file_paths[0] and loaders/val/file_paths[0] to the generated h5 folder or h5_aligned for the second model. 

TODOOOO Noch Prediction Calls reinpacken

