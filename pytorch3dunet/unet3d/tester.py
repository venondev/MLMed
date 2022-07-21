import time

from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from pytorch3dunet.unet3d import utils
from pytorch3dunet.unet3d.metrics import MedMl, MedMlSub
from pytorch3dunet.unet3d.utils import get_logger
import torch
import numpy as np
import torch.nn as nn
import os

import h5py

from scipy import ndimage as ndi
import nibabel as nib

logger = get_logger('UNetTester')




class PrecomputedTester():

    def __init__(self, precomputed_path_aligned, precomputed_path_normal, original_path, final_out_path,evaluate):
        self.metric = MedMlSub()
        self.evaluate=evaluate
        self.val_scores = utils.EvalScoreTracker()
        self.precomputed_path_aligned = precomputed_path_aligned
        self.precomputed_path_normal = precomputed_path_normal
        self.original_path = original_path
        self.final_out_path=final_out_path

    def load_aligned(self, file):
        if self.precomputed_path_aligned is None:
            return None
        sum_ = nib.load(os.path.join(self.precomputed_path_aligned, file + "_sum_rescaled.nii.gz")).get_fdata()
        return sum_ / 3

    def load_label(self, file):
        label_nifti = nib.load(os.path.join(self.original_path, file + "_masks.nii.gz"))

        return label_nifti, label_nifti.get_fdata()

    def load_raw(self, file):
        label_nifti = nib.load(os.path.join(self.original_path, file + "_orig.nii.gz"))

        return label_nifti, label_nifti.get_fdata()

    def load_normal(self, file):
        if self.precomputed_path_normal is None:
            return None
        sum_ = nib.load(os.path.join(self.precomputed_path_normal, file + "_pred.nii.gz")).get_fdata()
        dev_ = nib.load(os.path.join(self.precomputed_path_normal, file + "_dev.nii.gz")).get_fdata() + 0.1e-10
        return (sum_ / dev_)

    def evaluate(self):
        logger.info(
            f"precomputed_path_aligned: {self.precomputed_path_aligned}, precomputed_path_normal: {self.precomputed_path_normal}, original_path: {self.original_path} ")
        list_path = self.precomputed_path_normal if self.precomputed_path_normal is not None else self.precomputed_path_aligned
        files = os.listdir(list_path)
        files = list(filter(lambda x: x.endswith(".nii.gz"), files))
        if self.precomputed_path_normal is not None:
            files = list(map(lambda x: "_".join(x.split("_")[:-1]), files))
        else:
            files = list(map(lambda x: "_".join(x.split("_")[:-2]), files))
        files = list(set(files))
        if not os.path.exists("./final_val"):
            os.makedirs("./final_val")

        for file in tqdm(files):
            print(file)
            if not self.evaluate:
                label_nifti, label = self.load_raw(file)
            else:
                label_nifti, label = self.load_label(file)
            ratio = label_nifti.header.get("pixdim")[1:4].mean()
            print(ratio)

            div = 0
            aligned_pred = self.load_aligned(file)
            normal_pred = self.load_normal(file)
            sum = np.zeros_like(label, dtype='float64')
            label = label

            if self.precomputed_path_normal is not None:
                sum += normal_pred
                div += 1
            if self.precomputed_path_aligned is not None:
                sum += aligned_pred
                div += 1

            pred = (sum / div) > 0.5
            # pred,_ = self.calc_single_aneus_pred(pred)
            pred[pred > 0] = 1

            nib.save(nib.Nifti1Image(pred, label_nifti.affine, header=label_nifti.header),
                     self.final_out_path+'/' + file + '_pred.nii.gz')

            if self.evaluate:
                eval_score = self.metric(torch.tensor(pred[np.newaxis, np.newaxis]),
                                         torch.tensor(label[np.newaxis, np.newaxis]), ratio=ratio)
                self.val_scores.update(eval_score, 1)

    def calc_single_aneus_pred(self, pred, threshold=60):
        pred_labeled, num_single_aneus_pred = ndi.label(pred)

        keep = []
        for aneu_idx in range(1, num_single_aneus_pred + 1):
            cur = pred_labeled == aneu_idx
            if cur.sum() <= threshold:
                logger.info(f"Filtered out")
                pred_labeled[cur] = 0
            else:
                keep.append(aneu_idx)

        for idx, aneu_idx in enumerate(keep):
            pred_labeled[pred_labeled == aneu_idx] = idx + 1

        return pred_labeled, len(keep)


class Tester:
    def __init__(self, model, device, test_out_path, **kwargs):
        self.model = model
        self.device = device
        self.metric = MedMl()
        self.val_scores = utils.EvalScoreTracker()
        self.test_out_path = test_out_path

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
        test = True
        # Run predictions on the entire input dataset

        with torch.no_grad():
            k = 0
            start = time.time()
            for batch, indices in tqdm(test_loader, ncols=70, unit="Slices"):
                # position of slice
                k += 1

                batch = batch.to(self.device)
                predictions = self.final_activation(self.model(batch))
                center_size = torch.tensor(predictions[0].squeeze().size()) / 2
                padding = (center_size / 2).int().tolist()
                weight = torch.nn.functional.pad(torch.ones((center_size).int().tolist()), (
                    padding[0], padding[0], padding[1], padding[1], padding[2], padding[2]), 'constant', value=0).to(
                    self.device)
                for idx, pred in zip(indices, predictions):
                    result[idx] += pred.squeeze() * weight
                    dev[idx] += torch.ones_like(result[idx]) * weight
            time_dif = time.time() - start

        # Save results to disk
        test_path_split = test_loader.dataset.file_path.split('/')
        name = test_path_split[-1].replace(".h5", "")
        orig_path = "/".join(test_path_split[:-2])



        orig_data = nib.load(orig_path + "/" + name + "_orig.nii.gz")
        nib.save(nib.Nifti1Image(result.cpu().numpy(), orig_data.affine, header=orig_data.header),
                 self.test_out_path + '/' + name + '_pred.nii.gz')
        nib.save(nib.Nifti1Image(dev.cpu().numpy(), orig_data.affine, header=orig_data.header),
                 self.test_out_path + '/' + name + '_dev.nii.gz')

        return (time_dif, name)
