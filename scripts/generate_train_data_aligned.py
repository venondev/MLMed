import threading

import h5py
import os
import numpy as np
import scipy.ndimage as ndi
import sys
from utils import normalize, create_folder_if_not_exist, batch

# Calc stats Max Aneurysm Size
THRESH = 16


def get_bounds(label):
    x_s = label.sum(axis=(1, 2))
    y_s = label.sum(axis=(0, 2))
    z_s = label.sum(axis=(0, 1))

    x = np.where(x_s)[0][[0, -1]]
    y = np.where(y_s)[0][[0, -1]]
    z = np.where(z_s)[0][[0, -1]]

    return x, y, z


def get_sizes(label):
    x, y, z = get_bounds(label)
    x_size = x[1] - x[0]
    y_size = y[1] - y[0]
    z_size = z[1] - z[0]

    return np.array([x_size, y_size, z_size])


def transform_case(f_path_in, f_path_out):
    with h5py.File(f_path_in, "r") as f:
        raw = f["raw"][:]
        mask = f["label"][:]
        artery = f["artery"][:]
        overlap_mask = f["overlap_mask"][:]

        fac_sizes = []
        t2, num_labels_mask = ndi.label(mask)
        for i in range(1, num_labels_mask + 1):
            # Volume and overlap
            cur_mask = t2 == i

            max_size = get_sizes(cur_mask).max()
            print("Max", max_size)

            for fac in [1.5, 1.25, 1, 0.75, 0.5, 0.3]:
                if max_size * fac < THRESH:
                    fac_sizes.append(fac)
                    break

        print("fac", fac_sizes)
        fac_size = np.unique(fac_sizes)[0]
        if len(np.unique(fac_sizes)) > 1:
            print(f"Different sizes: {np.unique(fac_sizes)}")
            fac_size = 1

        if fac_size != 1:
            raw = ndi.zoom(raw, (fac_size, fac_size, fac_size), order=3)
            mask = ndi.zoom(mask, (fac_size, fac_size, fac_size), order=0)
            mask = mask > 0.5
            artery = ndi.zoom(artery, (fac_size, fac_size, fac_size), order=0)
            artery = artery > 0.5
            overlap_mask = ndi.zoom(overlap_mask, (fac_size, fac_size, fac_size), order=0)
            overlap_mask = overlap_mask > 0.5

        # Normalize
        raw = normalize(raw)

        with h5py.File(f_path_out, "w") as f_out:
            f_out.create_dataset("raw", data=raw)
            f_out.create_dataset("label", data=mask)
            f_out.create_dataset("artery", data=artery)
            f_out.create_dataset("overlap_mask", data=overlap_mask)

        return fac_size


def generate_data(datapath, type):
    out_path = os.path.join(datapath, type,  "h5_aligned")
    in_path = os.path.join(datapath, type, "h5")
    files = os.listdir(in_path)
    print(files)

    create_folder_if_not_exist(out_path)

    for b_idx, b in enumerate(batch(files, 8)):
        print(f"Batch {b_idx} / {len(files) // 8 if len(files) % 8 == 0 else len(files) // 8 + 1}")

        threads = []
        for idx, name in enumerate(b):
            if not name.endswith(".h5"):
                continue

            t = threading.Thread(target=transform_case, args=(
                os.path.join(in_path, name),
                os.path.join(out_path, name)))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()


INPUT_PATH = sys.argv[1] if len(sys.argv) == 2 else "/media/lm/Samsung_T5/Uni/Medml/t"

if __name__ == "__main__":
    generate_data(INPUT_PATH, "train")
    generate_data(INPUT_PATH, "val")
