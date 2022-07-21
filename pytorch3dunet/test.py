import importlib
import json
import os

import torch
import torch.nn as nn
from tqdm import tqdm

from pytorch3dunet.datasets.utils import get_test_loaders
from pytorch3dunet.unet3d import utils
from pytorch3dunet.unet3d.config import load_config
from pytorch3dunet.unet3d.model import get_model
from pytorch3dunet.unet3d.tester import Tester, PrecomputedTester

logger = utils.get_logger('UNet3DTest')


def main():
    # Load configuration
    config, _, is_test = load_config()
    test_out_path = config.get("test_out_path", "./test_out/")


    # Create the model

    model = get_model(config['model'])

    # Load model state
    model_path = config['model_path']
    logger.info(f'Loading model from {model_path}...')
    utils.load_checkpoint(model_path, model)
    # use DataParallel if more than 1 GPU available
    device = config['device']
    if torch.cuda.device_count() > 1 and not device.type == 'cpu':
        model = nn.DataParallel(model)
        logger.info(f'Using {torch.cuda.device_count()} GPUs for prediction')

    logger.info(f"Sending the model to '{device}'")
    model = model.to(device)

    # create predictor instance
    if not os.path.exists(test_out_path):
        os.makedirs(test_out_path)
    tester = Tester(model, device, test_out_path)
    test_loaders = get_test_loaders(config)
    files = os.listdir(config["loaders"]["test"]["file_paths"][0])
    files = list(filter(lambda x: x.endswith(".h5"), files))
    num_of_files = len(files)
    ptimes = []
    for t, test_loader in enumerate(test_loaders):
        # run the model prediction on the test_loader and save the results in the output_dir
        logger.info(f"[{t}/{num_of_files}]")
        p_time, name = tester(test_loader)
        ptimes.append({"id": name, "processing_time_in_seconds": p_time})

    with open("./test_out/p_time.json", "w") as outfile:
        json.dump(ptimes, outfile, indent=4)



if __name__ == '__main__':
    main()
