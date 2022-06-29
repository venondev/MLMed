import importlib
import os
import wandb
import torch.nn as nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from pytorch3dunet.unet3d.utils import DefaultTensorboardFormatter


class OnlineLogger:
    def __init__(self, model, config):
        raise NotImplementedError

    def log_stats(self, loss, eval, eval_detailed, step, prefix):
        raise NotImplementedError

    def log_params(self, step):
        raise NotImplementedError

    def log_images_upload(self, step, prefix):
        raise NotImplementedError

    def log_learning_rate(self, lr, step):
        raise NotImplementedError

    def log_model(self, is_best, metadata):
        raise NotImplementedError

    def log_images(self, input, target, prediction, step, prefix):
        raise NotImplementedError

    def log_non(self, step):
        raise NotImplementedError


class DisableLogger:
    def __init__(self, model, config):
        return

    def log_stats(self, loss, eval, eval_detailed, step, prefix):
        return

    def log_params(self, step):
        return

    def log_images_upload(self, step, prefix):
        return

    def log_learning_rate(self, lr, step):
        return

    def log_model(self, is_best, metadata):
        return

    def log_images(self, input, target, prediction, step, prefix):
        return

    def log_non(self, step):
        return


class TensorboardLogger(OnlineLogger):
    def __init__(self, model, config):
        self.model = model
        self.writer = SummaryWriter(log_dir=os.path.join(config["trainer"]["checkpoint_dir"], 'logs'))
        self.tensorboard_formatter = get_tensorboard_formatter(config["trainer"].pop('tensorboard_formatter', None))
        self.log_model_params = config["trainer"]["log_model_params"]

    def log_stats(self, loss, eval, eval_detailed, step, prefix):
        tag_value = {
            f'{prefix}_loss_avg': loss,
            f'{prefix}_eval_score_avg': eval
        }

        for tag, value in tag_value.items():
            self.writer.add_scalar(tag, value, step)

    def log_params(self, step):
        if self.log_model_params:
            for name, value in self.model.named_parameters():
                self.writer.add_histogram(name, value.data.cpu().numpy(), step)
                self.writer.add_histogram(name + '/grad', value.grad.data.cpu().numpy(), step)

    def log_model(self, is_best, metadata):
        return

    def log_learning_rate(self, lr, step):
        self.writer.add_scalar('learning_rate', lr, step)

    def log_images_upload(self, step, prefix):
        return

    def log_images(self, input, target, prediction, step, prefix):
        if self.model.training:
            if isinstance(self.model, nn.DataParallel):
                net = self.model.module
            else:
                net = self.model

            if net.final_activation is not None:
                prediction = net.final_activation(prediction)

        inputs_map = {
            'inputs': input,
            'targets': target,
            'predictions': prediction
        }
        img_sources = {}
        for name, batch in inputs_map.items():
            if isinstance(batch, list) or isinstance(batch, tuple):
                for i, b in enumerate(batch):
                    img_sources[f'{name}{i}'] = b.data.cpu().numpy()
            else:
                img_sources[name] = batch.data.cpu().numpy()

        for name, batch in img_sources.items():
            for tag, image in self.tensorboard_formatter(name, batch):
                self.writer.add_image(prefix + tag, image, step)

    def log_non(self, step):
        return


def get_tensorboard_formatter(formatter_config):
    if formatter_config is None:
        return DefaultTensorboardFormatter()

    class_name = formatter_config['name']
    m = importlib.import_module('pytorch3dunet.unet3d.utils')
    clazz = getattr(m, class_name)
    return clazz(**formatter_config)


class WandBLogger(OnlineLogger):
    def __init__(self, model, config):
        self.model = model
        wandb.init(project="aneurysm-segmentation", entity="mlmed", config=config, name=config["run"]["name"],
                   notes=config["run"]["notes"])
        self.log_model_params = config["trainer"]["log_model_params"]

        if self.log_model_params:
            wandb.watch(model, log_freq=config["trainer"]["log_after_iters"])
        self.file_path = config["trainer"]["checkpoint_dir"] + "/last_checkpoint.pytorch"
        self.temp_images = []

    def log_stats(self, loss, eval, eval_detailed, step, prefix):
        log_dict = {
            f"{prefix}_loss": loss,
            f"{prefix}_eval_score": eval,
        }

        for key in eval_detailed:
            log_dict[f"detailed/{prefix}_{key}"] = eval_detailed[key]

        wandb.log(log_dict, step=step)

    def log_model(self, is_best, metadata):
        art = wandb.Artifact(f'med-nn-{wandb.run.id}', type="model", metadata=metadata)
        art.add_file(self.file_path, "model.pytorch", is_tmp=True)

        if is_best:
            wandb.log_artifact(art, aliases=["latest", "best"])

        else:
            pass
            # wandb.log_artifact(art)

    def log_params(self, step):
        return

    def log_learning_rate(self, lr, step):
        wandb.log({"learning_rate": lr}, step=step)

    def log_images_upload(self, step, prefix):

        wandb.log({prefix + "_images": self.temp_images}, step=step)
        self.temp_images = []

    def log_images(self, input, target, prediction, step, prefix):
        if self.model.training:
            if isinstance(self.model, nn.DataParallel):
                net = self.model.module
            else:
                net = self.model

            if net.final_activation is not None:
                prediction = net.final_activation(prediction)

        inputs_map = {
            'inputs': input,
            'targets': target,
            'predictions': prediction
        }
        img_sources = {}
        for name, batch in inputs_map.items():
            img_sources[name] = batch.data.cpu().numpy()

        img_end = {}
        z = np.squeeze(img_sources["targets"])

        slice_idx = z.shape[1] // 2
        an = np.asarray((np.where(z == 1)))
        if an.shape[1] > 0:
            p = an[1, an.shape[1] // 2]
            slice_idx = p
        for name, batch in img_sources.items():
            img_end[name] = np.squeeze(batch)[:, slice_idx]

        class_labels = {
            1: "aneurysm"
        }

        masked_image = wandb.Image(img_end["inputs"], masks={
            "predictions": {
                "mask_data": np.where(img_end["predictions"] > 0.5, 1, 0),
                "class_labels": class_labels
            },
            "ground_truth": {
                "mask_data": img_end["targets"],
                "class_labels": class_labels
            }
        })
        self.temp_images.append(masked_image)

    def log_non(self, step):
        wandb.log({"_steps_": step}, step=step)


def get_class(class_name):
    m = importlib.import_module(__name__)
    clazz = getattr(m, class_name, None)
    if clazz is not None:
        return clazz
    raise RuntimeError(f'VisualLogger class: {class_name}')
