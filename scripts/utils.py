import numpy as np
import scipy.ndimage as ndi
import os


def normalize(raw):
    min_val = raw.min()
    max_val = raw.max()

    raw = (raw - min_val) / (max_val - min_val)

    return raw


def get_vessel_segmentation(raw_np, volume_threshold=2000, closing_thres=3):
    # Find Artery
    perc = np.percentile(raw_np, 99)

    seg = (raw_np > perc).astype(int)

    seg_labeled, num_labels = ndi.label(seg)
    unique, counts = np.unique(seg_labeled, return_counts=True)

    keep_idx = unique[counts > volume_threshold]
    keep = None
    for i in keep_idx:
        if i == 0:
            continue

        if keep is None:
            keep = seg_labeled == i
        else:
            keep = np.logical_or(keep, (seg_labeled == i))

    remove_idx = np.logical_not(keep)

    seg[remove_idx] = 0

    if closing_thres > 0:
        seg = ndi.binary_closing(seg, iterations=closing_thres)

    return seg


def get_case_names(datapath):
    files = os.listdir(datapath)
    files = filter(lambda x: x.endswith("_orig.nii.gz"), files)
    files = list(map(lambda x: x.replace("_orig.nii.gz", ""), files))
    sorted(files)

    return files


def create_folder_if_not_exist(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


def batch(iterable, n=1):
    len_it = len(iterable)
    for ndx in range(0, len_it, n):
        yield iterable[ndx:min(ndx + n, len_it)]