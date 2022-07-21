import os

import h5py
from torch import nn as nn
import torch

pool = nn.MaxPool3d((2, 2, 2))
for dataset_type in ["train", "val"]:

    # Input Dir
    p = f"./data/{dataset_type}"
    files = os.listdir(p)

    for idx, file in enumerate(files):
        print(f"{idx}/{len(files)}: {file}")
        with h5py.File(f"{p}/{file}", "r+") as f:
            mask = f["label"][:]
            raw = f["raw"][:]

            # raw = pool(torch.from_numpy(raw[None, None]))[0, 0].numpy()
            # mask = pool(torch.from_numpy(mask[None, None]))[0, 0].numpy()

            # Normalize
            min_val = raw.min()
            max_val = raw.max()

            raw = (raw - min_val) / (max_val - min_val)

            f["raw"][...] = raw
