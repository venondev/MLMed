import importlib
import json
import os
import threading
import time

import torch
import torch.nn as nn
import h5py
import scipy.ndimage as ndi
import nibabel as nib
import numpy as np
from tqdm import tqdm

from pytorch3dunet.datasets.utils import get_test_loaders
from pytorch3dunet.unet3d import utils
from pytorch3dunet.unet3d.config import load_config
from pytorch3dunet.unet3d.model import get_model
from pytorch3dunet.unet3d.tester import Tester, PrecomputedTester

logger = utils.get_logger('UNet3DTest')

multisize = True

multisize_path = "multisize"
multisize_pred_path = "multisize_pred"
multisize_rescaled_path = "multisize_rescaled"

def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


def resize(raw, fac, order=3):
    if fac != 1:
        zoomed = ndi.zoom(raw, (fac, fac, fac), order=order)
    else:
        zoomed = raw
    return zoomed

mutex = threading.Lock()


def convert(times, factor, file, datapath_out, raw, cur_idx, files):
    file_path = datapath_out + "/" + file + "_" + str(factor) + ".h5"
    if os.path.exists(file_path):
        return

    start = time.time()
    zoomed = resize(raw, factor, order=3)
    with h5py.File(file_path, "w") as f:
        f.create_dataset("raw", data=zoomed)
    print(f"{factor:} {cur_idx + 1}/{len(files)} : {file}")
    t_diff = time.time() - start
    mutex.acquire()
    times[file] += t_diff
    mutex.release()


def resize_multisize(times, raw_path, save_path, files):

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for b_idx, b_batch in enumerate(batch(files, 22)):

        files_conv = []
        for idx, file in enumerate(b_batch):
            if file.endswith(".nii.gz"):
                start = time.time()
                raw = nib.load(raw_path + "/" + file).get_fdata()

                # normalize
                min_val = raw.min()
                max_val = raw.max()
                raw = (raw - min_val) / (max_val - min_val)

                file = file.replace('_orig.nii.gz', '')
                files_conv.append((file, raw))
                print(f"{idx + 1}/{len(b_batch)} : {file}")

                # First Time writeing to times
                mutex.acquire()
                times[file] = time.time() - start
                mutex.release()

        print("Loaded files in mem")

        for fac in [1.5, 1.25, 1, 0.75, 0.5, 0.3]:
            threads = []
            for idx, (file, raw) in enumerate(files_conv):
                t = threading.Thread(target=convert, args=(times, fac, file, save_path, raw, idx, files))
                t.start()

                threads.append(t)

            for t in threads:
                t.join()


def transform_case(times, raw_path, pred_path, out_path, idx, file, files):
    print(f"{idx}/{len(files)}: {file}")

    raw = nib.load(raw_path + "/" + file + "_orig.nii.gz")

    raw_shape = raw.shape

    start = time.time()
    sum_ = np.zeros(raw_shape)
    for fac in [1.5, 1.25, 1, 0.75, 0.5, 0.3]:
        print("factor: ", fac)
        dev_raw = nib.load(os.path.join(pred_path, file + "_" + str(fac) + "_dev.nii.gz"))
        dev = dev_raw.get_fdata()
        pred = nib.load(os.path.join(pred_path, file + "_" + str(fac) + "_pred.nii.gz")).get_fdata()

        pred = np.divide(pred, dev, out=np.zeros_like(pred), where= dev != 0)
        print("pred shape: ", pred.shape, pred.min(), pred.max())

        pred_zoom = resize(pred, 1 / fac, order=3)

        min_shape = np.minimum(raw_shape, pred_zoom.shape)
        sum_[:min_shape[0], :min_shape[1], :min_shape[2]] += pred_zoom[:min_shape[0], :min_shape[1], :min_shape[2]]

    print("Sum", sum_.dtype, sum_.shape, sum_.min(), sum_.max())
    nib.save(nib.Nifti1Image(sum_, raw.affine, raw.header), os.path.join(out_path, file + "_sum_rescaled.nii.gz"))
    diff = time.time() - start
    mutex.acquire()
    times[file] += diff
    mutex.release()


def rescale(times, raw_path, pred_path, out_path):

    files = os.listdir(pred_path)
    files = ["_".join(f.replace("_dev.nii.gz", "").split("_")[:-1]) for f in files if f.endswith("_dev.nii.gz")]
    files = list(set(files))

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    for b_idx, b_batch in enumerate(batch(files, 8)):
        threads = []
        for idx, file in enumerate(b_batch):
            t = threading.Thread(target=transform_case, args=(times, raw_path, pred_path, out_path, 8 * b_idx + idx, file, files))
            t.start()
            threads.append(t)

        for t in threads:
            t.join()
        print(f"Batch {b_idx + 1}/{len(files) // 8}")

def main():
    # Load configuration
    config, _, is_test = load_config()

    file_path = config["loaders"]["test"]["file_paths"][0]
    files = os.listdir(file_path)
    files = list(filter(lambda x: x.endswith("orig.nii.gz"), files))

    if False:
        print("Using precomputed predictions")
        tester = PrecomputedTester(file_path)
        tester.evaluate()
        # TODO: Schauen wie wir das mit der Time machen
        eval_score, eval_score_detailed = tester.val_scores.avg
        if hasattr(tester.metric, "compute_final"):
            eval_score, eval_score_detailed = tester.metric.compute_final(tester.val_scores)
        print(eval_score, eval_score_detailed)
    else:
        # Create the model

        times = {}
        # Generate Multiscale Inputs
        multisize_save_path = file_path + "/" + multisize_path
        raw_path = file_path
        resize_multisize(times, raw_path, multisize_save_path, files)
        print("Finished Scaling...")
        print(times)

        model = get_model(config['model'])

        # Load model state
        model_path = config['model_path']
        logger.info(f'Loading model from {model_path}...')
        utils.load_checkpoint(model_path, model)
        # use DataParallel if more than 1 GPU available
        device = config['device']
        if torch.cuda.device_count() > 1 and not device.type == 'cpu':
            model = nn.DataParallel(model)
            logger.info(f'Using {torch.cuda.device_count()} GPUs for prediction')

        logger.info(f"Sending the model to '{device}'")
        model = model.to(device)

        # create predictor instance
        tester = Tester(model, device)

        config["loaders"]["test"]["file_paths"][0] = multisize_save_path
        test_loaders = get_test_loaders(config)
        pred_path = file_path + "/test_out"

        files = os.listdir(multisize_save_path)
        files = list(filter(lambda x: x.endswith(".h5"), files))
        num_of_files = len(files)
        for t, test_loader in enumerate(test_loaders):
            # run the model prediction on the test_loader and save the results in the output_dir
            logger.info(f"[{t}/{num_of_files}]")
            p_time, name = tester(test_loader, pred_path)

            times[name] += p_time
        print("Finished Prediction")
        print(times)

        # Rescale the predictions
        out_path = file_path + "/test_out_rescaled"
        rescale(times, file_path, pred_path, out_path)
        print("Finished Rescale")
        print(times)

        if True:
            print("Testing")
            tester = PrecomputedTester(out_path)
            tester.evaluate()

            eval_score, eval_score_detailed = tester.val_scores.avg
            if hasattr(tester.metric, "compute_final"):
                eval_score, eval_score_detailed = tester.metric.compute_final(tester.val_scores)
            print(eval_score, eval_score_detailed)

        print(times)
        with open(f"{file_path}/p_time.json", "w") as outfile:
            json.dump(times, outfile, indent=4)


if __name__ == '__main__':
    main()
