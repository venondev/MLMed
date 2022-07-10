import time

from tqdm import tqdm

from pytorch3dunet.unet3d import utils
from pytorch3dunet.unet3d.metrics import MedMl
from pytorch3dunet.unet3d.utils import get_logger
import torch
import numpy as np
import torch.nn as nn
import os

import h5py

from scipy import ndimage as ndi
import nibabel as nib

logger = get_logger('UNetTester')


class Tester:
    def __init__(self, model, device, **kwargs):
        self.model = model
        self.device = device
        self.metric = MedMl()
        self.val_scores = utils.EvalScoreTracker()

    def final_activation(self, input):
        if isinstance(self.model, nn.DataParallel):
            fa = self.model.module.final_activation
        else:
            fa = self.model.final_activation
        if fa is not None:
            return fa(input)
        else:
            input

    def __call__(self, test_loader):
        result = torch.zeros(test_loader.dataset.raw_full_shape).to(self.device)
        dev = torch.zeros(test_loader.dataset.raw_full_shape).to(self.device)
        self.model.eval()
        # Run predictions on the entire input dataset
        with torch.no_grad():
            k = 0
            start = time.time()
            for batch, indices in tqdm(test_loader, ncols=70, unit="Slices"):
                # position of slice
                k += 1

                batch = batch.to(self.device)
                predictions = self.final_activation(self.model(batch))
                for idx, pred in zip(indices, predictions):
                    result[idx] += pred.squeeze()
                    dev[idx] += torch.ones_like(result[idx])
            time_dif = time.time() - start

        # Save results to disk
        test_path_split = test_loader.dataset.file_path.split('/')
        name = test_path_split[-1].replace(".h5", "")
        orig_path = "/".join(test_path_split[:-2])

        if not os.path.exists("./test_out"):
            os.makedirs("./test_out")

        orig_data = nib.load(orig_path + "/" + name + "_masks.nii.gz")
        nib.save(nib.Nifti1Image(result.cpu().numpy(), orig_data.affine, header=orig_data.header),
                 './test_out/' + name + '_pred.nii.gz')
        nib.save(nib.Nifti1Image(dev.cpu().numpy(), orig_data.affine, header=orig_data.header),
                 './test_out/' + name + '_dev.nii.gz')

        result /= dev
        result[result >= 0.4] = 1
        result[result < 1] = 0

        label = h5py.File(test_loader.dataset.file_path, 'r')[test_loader.dataset.label_internal_path][:]
        eval_score = self.metric(result[np.newaxis, np.newaxis].cpu(), torch.tensor(label[np.newaxis, np.newaxis]))
        self.val_scores.update(eval_score, 1)
        return (time_dif, name)
