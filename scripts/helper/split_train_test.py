import os
import shutil
import json


def move(files_f, input_path, output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path, mode=0o777)

    for i in files_f:
        shutil.copyfile(os.path.join(input_path, i + "_orig.nii.gz"), os.path.join(output_path, i + "_orig.nii.gz"))
        shutil.copyfile(os.path.join(input_path, i + "_masks.nii.gz"), os.path.join(output_path, i + "_masks.nii.gz"))


def split_train_test_from_file(json_path, input_path, output_path):
    # Opening JSON file
    with open(json_path, "r") as f:
        # returns JSON object as a dictionary
        data = json.load(f)
        train = data["train"]
        val = data["val"]

        move(train, input_path, os.path.join(output_path, "train"))
        move(val, input_path, os.path.join(output_path, "val"))

JSON_PATH = "../data/train_test_split.json"
INPUT_PATH = "/media/lm/Samsung_T5/Uni/Medml/t"
OUTPUT_PATH = "/media/lm/Samsung_T5/Uni/Medml/t"

if __name__ == "__main__":
    split_train_test_from_file(JSON_PATH, INPUT_PATH, OUTPUT_PATH)