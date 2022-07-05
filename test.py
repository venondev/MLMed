"""
Script to demonstrate the usage of shared arrays using multiple workers.
In the first epoch the shared arrays in the dataset will be filled with
random values. After setting set_use_cache(True), the shared values will be
loaded from multiple processes.
@author: ptrblck
"""

import torch
from torch.utils.data import Dataset, DataLoader

import ctypes
import multiprocessing as mp
import time
import numpy as np


class MyDataset(Dataset):
    def __init__(self):
        print("init")


    def __getitem__(self, index):
        time.sleep(1)
        return 1

    def __len__(self):
        return 10



dataset = MyDataset()
loader = DataLoader(
    dataset,
    num_workers=4,
    shuffle=False
)

for epoch in range(10):

    for idx, data in enumerate(loader):
        print('Epoch {}, idx {}, data.shape {}, data {}'.format(epoch, idx, data.shape, data.item()))
