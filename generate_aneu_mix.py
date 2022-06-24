import argparse
import os
import numpy as np
from pathlib import Path

import yaml
import h5py
from pytorch3dunet.augment.transforms import Transformer

parser = argparse.ArgumentParser(description='Mixing Augmentation')
parser.add_argument('--config', type=str, help='Config Augmentation', required=True)
parser.add_argument('--count', type=int, help='Number of examples to generate', required=True)
parser.add_argument('--out', type=str, help='Output directory', default="./data/augs")

args = parser.parse_args()
config = yaml.safe_load(open(args.config, 'r'))

transform_config = config["loaders"]["train"]["transformer"]
path = config["loaders"]["train"]["file_paths"][0]


transformer = Transformer(transform_config, {})
raw_transform = transformer.raw_transform()
label_transform = transformer.label_transform()

files = os.listdir(path)
id = np.random.randint(low=0, high=1000)

for i in range(args.count):
    print("Start")
    pick = np.random.choice(files)
    case = h5py.File(f"{path}/{pick}", "r")

    print("Loaded")
    img_trans = raw_transform(case["raw"][:])
    label_trans = label_transform(case["label"][:])

    print("Transformed")
    Path(args.out).mkdir(parents=True, exist_ok=True)

    target_file = os.path.join(f"{args.out}/{pick}_{id}_{i}.h5")
    f = h5py.File(target_file, mode="a")
    f.create_dataset("raw", data=img_trans)
    f.create_dataset("label", data=label_trans)

    print("Finished case")






