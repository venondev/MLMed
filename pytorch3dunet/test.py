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


    if config.get("precomputed_predictions", False):
        print("Using precomputed predictions")
        tester = PrecomputedTester(config.get("precomputed_path_hjalmar",None),config.get("precomputed_path_philipp",None),config.get("original_path",None))
        tester.evaluate()

        # TODO: Schauen wie wir das mit der Time machen
    else:
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
        tester = Tester(model, device)
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

    eval_score, eval_score_detailed = tester.val_scores.avg
    if hasattr(tester.metric, "compute_final"):
        eval_score, eval_score_detailed = tester.metric.compute_final(tester.val_scores)
    print(eval_score, eval_score_detailed)

if __name__ == '__main__':
    main()
