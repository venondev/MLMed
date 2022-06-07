import os
from matplotlib.transforms import Transform
from torchvision.io import read_image
from torch.utils.data import Dataset
import torch
import nibabel as nib
import numpy as np
import pytorch3dunet.augment.transforms as transforms

def calculate_stats(images):
    """
    Calculates min, max, mean, std given a list of ndarrays
    """
    # flatten first since the images might not be the same size
    flat = np.concatenate(
        [img.ravel() for img in images]
    )
    return {'pmin': np.percentile(flat, 1), 'pmax': np.percentile(flat, 99.6), 'mean': np.mean(flat),
            'std': np.std(flat)}

class NiftiImageDataset(Dataset):

    def __init__(self, config, global_normalization=True) -> None:
        self.datapath = config["file_paths"][0]

        self.cases, self.images, self.masks, self.affines = self.load_data()

        if global_normalization:
            stats = calculate_stats(self.images)
            print("Stats---------------", stats)
        else:
            stats = {'pmin': None, 'pmax': None, 'mean': None, 'std': None}

        self.transformer = transforms.Transformer(config["transformer"], stats)
        self.raw_transform = self.transformer.raw_transform()
        self.label_transform = self.transformer.label_transform()

    def __len__(self):
        return len(self.cases)
    
    def __getitem__(self, idx):

        image = self.images[idx]
        mask = self.masks[idx]

        image = self.raw_transform(image)
        mask = self.label_transform(mask)
        
        return image, mask
    
    def load_data(self):
        cases = os.listdir(self.datapath)
        cases = filter(lambda x: x.endswith("_orig.nii.gz"), cases)
        cases = list(map(lambda x: x.replace("_orig.nii.gz", ""), cases))

        images = []
        masks = []
        affines = []

        n = len(cases)
        for idx, case in enumerate(cases):
            if idx % 10 == 0:
                print(f"Loading data {idx} / {n}")


            image_raw = nib.load(f"{self.datapath}/{case}_orig.nii.gz")
            mask_raw = nib.load(f"{self.datapath}/{case}_masks.nii.gz")

            image = image_raw.get_fdata()
            mask = mask_raw.get_fdata()
            affine = image_raw.affine

            images.append(image)
            masks.append(mask)
            affines.append(affine)
        
        print("Finished loading data")
        
        return cases, images, masks, affines