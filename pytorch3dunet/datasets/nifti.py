import os
from matplotlib.transforms import Transform
from torchvision.io import read_image
from torch.utils.data import Dataset
import torch
import nibabel as nib

class NiftiImageDataset(Dataset):

    def __init__(self, datapath, transform=None) -> None:
        self.datapath = datapath
        self.transform = transform

        files = os.listdir(self.datapath)
        files = filter(lambda x: x.endswith("_orig.nii.gz"), files)
        files = list(map(lambda x: x.replace("_orig.nii.gz", ""), files))

        self.cases = files
    
    def __len__(self):
        return len(self.cases)
    
    def __getitem__(self, idx):
        case = self.cases[idx]

        image_raw = nib.load(f"{self.datapath}/{case}_orig.nii.gz")
        mask_raw = nib.load(f"{self.datapath}/{case}_masks.nii.gz")

        image = torch.from_numpy(image_raw.get_fdata())
        mask = torch.from_numpy(mask_raw.get_fdata())

        if self.transform:
            image, mask = self.transform(image, mask)
        
        return image[None, :].float(), mask[None, :].float()



    
