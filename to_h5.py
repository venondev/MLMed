import glob
import logging
import os
from datetime import datetime

import h5py
import nibabel as nib
import numpy as np
from pathlib import Path

# Path to new hdf5 training folder 
h5_training_path = "./data/training_conv/"


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


# Transform NII files to HDF5
def nii_to_h5(files):

    for file_orig, file_mask, file_labeled in files:
        # convert nii to numpy
        orig = np.array(nib.load(file_orig).get_fdata())
        mask = np.array(nib.load(file_mask).get_fdata())
        labelMask = np.array(nib.load(file_labeled).get_fdata())

        orig, mask, labelMask = align_dims(orig, mask, labelMask)

        # new file name 
        file_name = str(file_mask).split("/")[-1].split(".")[0]

        numpy_to_h5(orig, mask, labelMask, file_name)

    print("Finished converting data")

def numpy_to_h5(orig, mask, labelMask, file_name):
    target_file = os.path.join(h5_training_path + file_name + ".h5")
    f = h5py.File(target_file, mode="a")
    
    # save original file as raw (as required in pytorch3dunet)
    f.create_dataset("raw", data = orig)
    # save mask as label (as required in pytorch3dunet)
    f.create_dataset("label", data = mask)
    # save labeled mask datas as labelMask
    f.create_dataset("labelMask", data = labelMask)

# Path to NII data
data_path = Path('./data/training')

# Load files
files = sorted(list(data_path.glob('*.*')))
files_orig = list(filter(lambda file: 'orig.nii.gz' in str(file), files))
files_masks = list(filter(lambda file: 'masks.nii.gz' in str(file), files))
files_labeled = list(filter(lambda file: 'labeledMasks.nii.gz' in str(file), files))

# convert to HDF5
files = zip(files_orig, files_masks, files_labeled)
nii_to_h5(files)