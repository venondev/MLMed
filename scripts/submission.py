import json

import numpy as np

import nibabel as nib
import scipy
from itertools import product
from scipy import ndimage as ndi
import os
import scipy
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(description='Download Weights & Biases artifact.')
parser.add_argument('--in_dir', dest='in_dir', type=str, required=True)

args = parser.parse_args()

input_data_path = args.in_dir+"/"


def transform_affine(input, transformation):
    assert input.shape[0] == 3, "only 3d coordinates (input.shape = (3,:))"
    assert transformation.shape == (4, 4), "wrong affine transformation shape (transformation.shape = (4,4))"
    ones = np.ones((1, input.shape[1]))
    affine_input = np.vstack((input, ones))
    output = (transformation @ affine_input)[:3]
    return output


def calc_candidate_json(labeled_prediction, affine, name, ptime):
    dataset_item = {"dataset_id": name, "processing_time_in_seconds": ptime, "candidates": []}
    for label in np.unique(labeled_prediction):
        if label == 0:
            continue
        single_prediction = np.zeros_like(labeled_prediction)
        single_prediction[labeled_prediction == label] = 1

        data = np.asarray(np.where(single_prediction == 1))
        data = transform_affine(data, affine)

        means = data.mean(axis=1)
        position = means
        cleaned_data = (data.T - means).T
        cov = np.cov(cleaned_data)
        v, w = np.linalg.eig(cov)
        rotated_data = w @ cleaned_data
        wpinv = np.linalg.pinv(w)
        rotated_mins = rotated_data.min(axis=1)
        rotated_maxs = rotated_data.max(axis=1)
        extend = np.abs(rotated_mins) + np.abs(rotated_maxs)
        orthogonal_offset_vectors = wpinv
        object_oriented_bounding_box = {"extent": extend.tolist(),
                                        "orthogonal_offset_vectors": orthogonal_offset_vectors.tolist()}
        candidate = {"position": position.tolist(), "object_oriented_bounding_box": object_oriented_bounding_box}
        dataset_item["candidates"].append(candidate)
    return dataset_item




files = os.listdir(input_data_path)
files = list(filter(lambda x: x.endswith("_pred.nii.gz"), files))
names = [i.replace("_pred.nii.gz", "") for i in files][:4]
print(names)
if not os.path.exists("./submission/task1/"):
    os.makedirs("./submission/task1/")
if not os.path.exists("./submission/task2/"):
    os.makedirs("./submission/task2/")


result_task1 = {"username": "Emu", "task_1_results": []}
with open(input_data_path + "p_time.json", "r") as f:
    ptimes = json.load(f)
for name in tqdm(names):
    # load data
    tqdm.write(f"Processing: {name}...")
    prediction_nifti = nib.load(input_data_path + name + '_pred.nii.gz')
    original_nifti = nib.load("/home/tu-pirlet/test_data" + name + '_pred.nii.gz')

    binary_prediction = prediction_nifti.get_fdata()

    # prediction -> binary prediction
    prediction_idx = np.where(binary_prediction > 0.5)
    labeled_prediction, _ = ndi.label(binary_prediction)

    # store task2 output

    nib.save(nib.Nifti1Image(labeled_prediction, original_nifti.affine, header=original_nifti.header),
             './submission/task2/' + name + '_output.nii.gz')

    # calc task1
    ptime = ptimes[name]
    result_task1["task_1_results"].append(calc_candidate_json(labeled_prediction, original_nifti.affine, name, ptime))
print(f"Store Task1 in ./submission/task1/task1.json")
print(f"Store Task2 in ./submission/task2/*")
with open("./submission/task1/task1.json", "w") as outfile:
    json.dump(result_task1, outfile, indent=4)
