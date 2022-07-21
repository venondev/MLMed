
# Machine Learning in Medical Image Processing Segmentation Implementation

## Introduction

This implementation is based on the 3D-UNet implementation by Adrian Wolny which you can find [here](https://github.com/wolny/pytorch-3dunet). We added an attention layer, a custom dataloader, custom metrics, a new prediction reconstruction, multiple preprocessing steps and multi size input preprocessing.

## Installation

Create virtual environment and install requirements from requirements.txt

## Models

We developed two models. The models have the same underlying architecture, the main difference is that the first model uses the raw full image as input for the train patches, while the second model uses train patches where the aneurysm sizes are aligned.
#### checkpoints:
- aligned model: model/aligned_model.pytorch
- normal model: model/normal_model.pytorch
## How to train the model

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


For the first model use the ./train-config/normal_config.yaml config and for the second model the ./train-config/multisize_config.yaml config.
You need to adjust the file path in loaders/train/file_paths[0] and loaders/val/file_paths[0] to the generated h5 folder or h5_aligned for the second model. 

## How to make predictions

First use the following script to generate the test data:

```bash
python ./scripts/generate_test_data.py PATH_TO_TEST_DATA
```

Then you can use the following script to make predictions with the first model:

```bash
python ./start_test.py --config ./test-config/normal_config_test.yaml
```

You need to adjust the model path to point to the .pytorch file and the file path to the h5 folder of the test data.

For the second model you also need to adjust the model path but change the file path to the raw test folder.

```bash
python ./start_test_resize.py --config ./test-config/multisize_config_test.yaml
```

To combine the predictions of the two models, you need to adjust the precomputed/original paths.
To get the metrics, you need to change evaluate to true.

```bash
python ./start_test_merge.py --config ./test-config/merge_config.yaml
```

The final predictions are stored in ./final_out

## How to generate the submission files

To get the submission format based on the predictions, you need to run the submission.py script:

```bash
python ./script/submission.py --in_dir ./final_out
```


