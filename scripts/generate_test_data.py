import h5py
import os
import nibabel as nib
import sys
from utils import create_folder_if_not_exist, get_case_names, normalize


def generate_data(datapath):
    files = get_case_names(datapath)

    h5_save_folder = os.path.join(datapath, "h5")

    create_folder_if_not_exist(h5_save_folder)

    for idx, name in enumerate(files):
        print(f"{idx} / {len(files)}: {name}")

        raw_np = nib.load(os.path.join(datapath, name + "_orig.nii.gz")).get_fdata()

        raw_np = normalize(raw_np)

        with h5py.File(os.path.join(h5_save_folder, f"{name}.h5"), "w") as f:
            f.create_dataset("raw", data=raw_np)

INPUT_PATH = sys.argv[1] if len(sys.argv) == 2 else "/media/lm/Samsung_T5/Uni/Medml/training/test"

if __name__ == "__main__":
    generate_data(INPUT_PATH)
