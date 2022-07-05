import os

import h5py
from torch import nn as nn
import torch
import numpy as np
from  scipy import ndimage


k=4
kernel_row=(np.array(list(range(1,k))+list(range(k,0,-1)))+4)/(k+4)
print(kernel_row)
kernel_2d=[]
for i in kernel_row:
    kernel_2d.append(i*kernel_row)
kernel_2d=np.asarray(kernel_2d)
kernel_3d=[]
for i in kernel_row:
    kernel_3d.append(kernel_2d*i)

kernel_3d=np.asarray(kernel_3d)


def clean_data(m):
        t=np.ones_like(m)
        t[m<0.4]=0
        t=ndimage.convolve(t,kernel_3d)
        return t*m
for dataset_type in ["train", "val"]:

    # Input Dir
    p = f"./data/{dataset_type}"
    files = os.listdir(p)

    for idx, file in enumerate(files):
        print(f"{idx}/{len(files)}: {file}")
        with h5py.File(f"{p}/{file}", "r") as f:
            mask = f["label"][:]
            raw = f["raw"][:]

            # Normalize
            min_val = raw.min()
            max_val = raw.max()

            raw = (raw - min_val) / (max_val - min_val)
            #raw = clean_data(raw)

            t = f"./data/{dataset_type}_nrom"
            if not os.path.isdir(t):
                os.mkdir(t)

            with h5py.File(f"{t}/{file}", "w") as f2:
                f2.create_dataset("raw", data=raw)
                f2.create_dataset("label", data=mask)
