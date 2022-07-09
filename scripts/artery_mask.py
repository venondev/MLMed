
import h5py
import os
import numpy as np
import scipy.ndimage as ndi
import nibabel as nib
import shutil

import json
data_path = "/home/tu-pirlet/training"



# Load files
files = os.listdir(data_path)
files = filter(lambda x: x.endswith("_orig.nii.gz"), files)
files = list(map(lambda x: x.replace("_orig.nii.gz", ""), files))


# Opening JSON file
f = open('/home/tu-pirlet/MLMed/data.json', )

# returns JSON object as
# a dictionary
data = json.load(f)

train = data["train"]
val = data["val"]

def move(files_f, folder):

    if not os.path.exists(folder):
        os.makedirs(folder,mode=0o777)

    for i in files_f:
        shutil.copyfile(os.path.join(data_path, i + "_orig.nii.gz"), os.path.join(folder, i + "_orig.nii.gz"))
        shutil.copyfile(os.path.join(data_path, i + "_masks.nii.gz"), os.path.join(folder, i + "_masks.nii.gz"))

move(train, "/home/tu-pirlet/MLMed/data/full_new_train")
move(val, "/home/tu-pirlet/MLMed/data/full_new_val")

def test(datapath):
    files = os.listdir(datapath)
    files = filter(lambda x: x.endswith("_orig.nii.gz"), files)
    files = list(map(lambda x: x.replace("_orig.nii.gz", ""), files))
    sorted(files)

    h5_save_folder = os.path.join(datapath, "h5")

    if not os.path.exists(h5_save_folder):
        os.makedirs(h5_save_folder)

    # PARAMS
    zoom = False
    volume_threshold = 2000 #2000
    closing_thres = 3

    volumes = []
    for idx, name in enumerate(files):
        print(f"{idx} / {len(files)}: {name}")

        raw = nib.load(os.path.join(datapath, name + "_orig.nii.gz"))
        label = nib.load(os.path.join(datapath, name + "_masks.nii.gz"))

        raw_np = raw.get_fdata()
        label_np = label.get_fdata()

        if zoom:
            raw_np = ndi.zoom(raw_np, (0.5, 0.5, 0.5), order=3)
            label_np = ndi.zoom(label_np, (0.5, 0.5, 0.5), order=0)
            label_np = label_np > 0.5

        # Find Artery
        perc = np.percentile(raw_np, 99)

        t = (raw_np > perc).astype(int)

        tt, num_labels = ndi.label(t)
        unique, counts = np.unique(tt, return_counts=True)

        keep_idx = unique[counts > volume_threshold]
        keep = None
        for i in keep_idx:
            if i == 0:
                continue

            if keep is None:
                keep = tt == i
            else:
                keep = np.logical_or(keep, (tt == i))

        remove_idx = np.logical_not(keep)

        t[remove_idx] = 0

        t = ndi.binary_closing(t, iterations=closing_thres)

        overlap_mask = np.logical_and(label_np, t)

        # Normalize
        min_val = raw_np.min()
        max_val = raw_np.max()

        raw_np = (raw_np - min_val) / (max_val - min_val)

        with h5py.File(os.path.join(h5_save_folder, f"{name}.h5"), "w") as f:
            f.create_dataset("raw", data=raw_np)
            f.create_dataset("label", data=label_np)
            f.create_dataset("artery", data=t)
            f.create_dataset("overlap_mask", data=overlap_mask)


test("/home/tu-pirlet/MLMed/data/full_new_train")
test("/home/tu-pirlet/MLMed/data/full_new_val")
