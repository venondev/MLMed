import h5py
import os
import numpy as np
import nibabel as nib
import sys
from utils import create_folder_if_not_exist, get_case_names, normalize, get_vessel_segmentation

# PARAMS
volume_threshold = 2000
closing_thres = 3


def generate_data(datapath, type):
    datapath = os.path.join(datapath, type)
    files = get_case_names(datapath)

    h5_save_folder = os.path.join(datapath, "h5")

    create_folder_if_not_exist(h5_save_folder)

    for idx, name in enumerate(files):
        print(f"{idx} / {len(files)}: {name}")

        raw_np = nib.load(os.path.join(datapath, name + "_orig.nii.gz")).get_fdata()
        label_np = nib.load(os.path.join(datapath, name + "_masks.nii.gz")).get_fdata()

        vessel_seg = get_vessel_segmentation(raw_np, volume_threshold, closing_thres)
        overlap_mask = np.logical_and(label_np, vessel_seg)
        raw_np = normalize(raw_np)

        with h5py.File(os.path.join(h5_save_folder, f"{name}.h5"), "w") as f:
            f.create_dataset("raw", data=raw_np)
            f.create_dataset("label", data=label_np)
            f.create_dataset("artery", data=vessel_seg)
            f.create_dataset("overlap_mask", data=overlap_mask)

INPUT_PATH = sys.argv[1] if len(sys.argv) == 2 else "/media/lm/Samsung_T5/Uni/Medml/t"

if __name__ == "__main__":
    generate_data(INPUT_PATH, "train")
    generate_data(INPUT_PATH, "val")
