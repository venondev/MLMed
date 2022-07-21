import glob
import logging
import os
from datetime import datetime

import h5py
import nibabel as nib
import numpy as np
from pathlib import Path

# Path to raw data
data_path = Path('./data/training')


def align_dims(image, mask, labeled_mask):
    if image.shape[2] < 220:
        diff = 220 - image.shape[2]

        image = np.concatenate([image] + [image[:, :, -1][:, :, np.newaxis]] * diff, axis=2)
        mask = np.concatenate([mask] + [mask[:, :, -1][:, :, np.newaxis]] * diff, axis=2)
        labeled_mask = np.concatenate([labeled_mask] + [labeled_mask[:, :, -1][:, :, np.newaxis]] * diff, axis=2)

    if image.shape[2] > 220:
        image = image[:, :, :220]
        mask = mask[:, :, :220]
        labeled_mask = labeled_mask[:, :, :220]

    return image, mask, labeled_mask


# 70 30 split
# Tiefe umstellen

# Transform NII files to HDF5
def nii_to_h5(files, save_path):
    for name in files:
        # convert nii to numpy
        orig = np.array(nib.load(f"{data_path}/{name}_orig.nii.gz").get_fdata())
        mask = np.array(nib.load(f"{data_path}/{name}_masks.nii.gz").get_fdata())
        labelMask = np.array(nib.load(f"{data_path}/{name}_labeledMasks.nii.gz").get_fdata())

        orig, mask, labelMask = align_dims(orig, mask, labelMask)

        orig = np.transpose(orig, (2, 0, 1))
        mask = np.transpose(mask, (2, 0, 1))
        labelMask = np.transpose(labelMask, (2, 0, 1))

        orig /= orig.max()

        numpy_to_h5(orig, mask, labelMask, name, save_path)

    print("Finished converting data")


def numpy_to_h5(orig, mask, labelMask, file_name, save_path):
    target_file = os.path.join(save_path + file_name + ".h5")
    f = h5py.File(target_file, mode="a")

    # save original file as raw (as required in pytorch3dunet)
    f.create_dataset("raw", data=orig)
    # save mask as label (as required in pytorch3dunet)
    f.create_dataset("label", data=mask)
    # save labeled mask datas as labelMask
    f.create_dataset("labelMask", data=labelMask)


# Path to NII data

train_split = 0.7
# Load files
files = os.listdir(data_path)
files = filter(lambda x: x.endswith("_orig.nii.gz"), files)
files = list(map(lambda x: x.replace("_orig.nii.gz", ""), files))
np.random.shuffle(files)

split = int(len(files) * train_split)
train = files[:split]
val = files[split:]

print(len(train))
print(len(val))

# convert to HDF5
nii_to_h5(train, "./data/train/")
nii_to_h5(val, "./data/val/")
