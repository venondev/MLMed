import os

import h5py
import numpy as np
from scipy import ndimage as ndi


def get_aneurysm_center(mask):
    x_s = mask.sum(axis=(1, 2))
    y_s = mask.sum(axis=(0, 2))
    z_s = mask.sum(axis=(0, 1))

    x = int(np.where(x_s)[0][[0, -1]].mean())
    y = int(np.where(y_s)[0][[0, -1]].mean())
    z = int(np.where(z_s)[0][[0, -1]].mean())

    return x, y, z


for dataset_type in ["train", "val"]:

    # Input Dir
    p = f"./data/{dataset_type}_half"
    files = os.listdir(p)

    for idx, file in enumerate(files):
        print(f"{idx}/{len(files)}: {file}")
        with h5py.File(f"{p}/{file}", "r+") as f:
            mask = f["label"][:]
            raw = f["raw"][:]

            labels, num_labels = ndi.label(mask)

            choosen_coords = []
            for i in range(num_labels):
                cur = (labels == i + 1)
                center_coords = get_aneurysm_center(cur)
                choosen_coords.append(center_coords)

            idx = np.mgrid[:mask.shape[0], :mask.shape[1], :mask.shape[2]]
            ret = np.zeros(mask.shape)
            for pic in choosen_coords:

                d = np.sqrt(np.sum((idx - np.array(pic)[:, None, None, None]) ** 2, axis=0))
                d[d == 0] = 1e-10
                t2 = np.clip(1 / d, 0, 1)

                if ret is None:
                    ret = t2
                else:
                    ret = np.max([ret, t2], axis=0)

            f.create_dataset("weight", data=ret)
