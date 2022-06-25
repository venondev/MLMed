import random
from pytorch3dunet.datasets import own_hdf5_npz_lazy

import torch
from pathlib import Path
from pytorch3dunet.unet3d.config import load_config
from pytorch3dunet.unet3d.trainer import create_trainer
from pytorch3dunet.unet3d.utils import get_logger

logger = get_logger('TrainingSetup')


def main():
    # Load and log experiment configuration
    config,store_slices = load_config()
    logger.info(config)

    manual_seed = config.get('manual_seed', None)
    if manual_seed is not None:
        logger.info(f'Seed the RNG for all devices with {manual_seed}')
        logger.warning('Using CuDNN deterministic setting. This may slow down the training!')
        random.seed(manual_seed)
        torch.manual_seed(manual_seed)
        # see https://pytorch.org/docs/stable/notes/randomness.html
        torch.backends.cudnn.deterministic = True

    # create trainer
    if store_slices:
        logger.info(f'Load HDF5 files and store slices as npz ....')
        path = Path(config["loaders"]["train"]["temp_file_path"])
        path.mkdir(parents=True, exist_ok=True)
        path = Path(config["loaders"]["val"]["temp_file_path"])
        path.mkdir(parents=True, exist_ok=True)

        own_hdf5_npz_lazy.create_all_slices(config['loaders'], phase='train')
        own_hdf5_npz_lazy.create_all_slices(config['loaders'], phase='val')
        exit()
    else:
        trainer = create_trainer(config)
    # Start training
    trainer.fit()


if __name__ == '__main__':
    main()
