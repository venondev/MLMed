import wandb
import shutil
import os
import argparse
parser = argparse.ArgumentParser(description='Download Weights & Biases artifact.')
parser.add_argument('--dir', dest='dir', type=str, default="model/base_model.pytorch",
                    help='destination directionary')
parser.add_argument('--id', dest='id',type=str,required=True,
                    help='wand id/name')

args = parser.parse_args()
api = wandb.Api()

artifact = api.artifact(args.id)
artifact_dir = artifact.checkout()
artifact_dir+="/"+os.listdir(artifact_dir)[0]
print(args.dir)
print(artifact_dir)
shutil.move(artifact_dir, args.dir)