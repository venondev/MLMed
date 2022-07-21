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
    config, _, is_test = load_config()
    final_out_path = config.get("final_out_path", "./final_out/")

    if config.get("precomputed_predictions", False):
        print("Using precomputed predictions")

        tester = PrecomputedTester(config.get("precomputed_path_aligned", None),
                                   config.get("precomputed_path_normal", None), config.get("original_path", None),final_out_path,evaluate=config.get("evaluate", False))
        tester.evaluate()