import os

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

from pytorch3dunet.datasets.utils import get_train_loaders
import numpy as np
from pytorch3dunet.unet3d import online_logger
from pytorch3dunet.unet3d.losses import get_loss_criterion
from pytorch3dunet.unet3d.metrics import get_evaluation_metric
from pytorch3dunet.unet3d.model import get_model
from pytorch3dunet.unet3d.utils import get_logger, create_optimizer, \
    create_lr_scheduler, get_number_of_learnable_parameters
from . import utils

logger = get_logger('UNet3DTrainer')


def create_trainer(config, test_run=False):
    # Create the model
    model = get_model(config['model'])
    # use DataParallel if more than 1 GPU available
    device = config['device']
    if torch.cuda.device_count() > 1 and not device.type == 'cpu':
        model = nn.DataParallel(model)
        logger.info(f'Using {torch.cuda.device_count()} GPUs for training')

    # put the model on GPUs
    logger.info(f"Sending the model to '{config['device']}'")
    model = model.to(device)

    # Log the number of learnable parameters
    logger.info(f'Number of learnable params {get_number_of_learnable_parameters(model)}')

    # Create loss criterion
    loss_criterion = get_loss_criterion(config)
    # Create evaluation metric
    eval_criterion = get_evaluation_metric(config)

    # Create data loaders
    loaders = get_train_loaders(config, test_run=test_run)

    # Create the optimizer
    optimizer = create_optimizer(config['optimizer'], model)

    # Create learning rate adjustment strategy
    lr_scheduler = create_lr_scheduler(config.get('lr_scheduler', None), optimizer)

    trainer_config = config['trainer']
    web_logger = online_logger.get_class(trainer_config["online_logger"])(model, config)

    # Create trainer
    resume = trainer_config.pop('resume', None)
    pre_trained = trainer_config.pop('pre_trained', None)

    return UNet3DTrainer(model=model,
                         optimizer=optimizer,
                         lr_scheduler=lr_scheduler,
                         loss_criterion=loss_criterion,
                         eval_criterion=eval_criterion,
                         web_logger=web_logger,
                         device=config['device'],
                         loaders=loaders,
                         resume=resume,
                         pre_trained=pre_trained,
                         **trainer_config)


class UNet3DTrainer:
    """3D UNet trainer.

    Args:
        model (Unet3D): UNet 3D model to be trained
        optimizer (nn.optim.Optimizer): optimizer used for training
        lr_scheduler (torch.optim.lr_scheduler._LRScheduler): learning rate scheduler
            WARN: bear in mind that lr_scheduler.step() is invoked after every validation step
            (i.e. validate_after_iters) not after every epoch. So e.g. if one uses StepLR with step_size=30
            the learning rate will be adjusted after every 30 * validate_after_iters iterations.
        loss_criterion (callable): loss function
        eval_criterion (callable): used to compute training/validation metric (such as Dice, IoU, AP or Rand score)
            saving the best checkpoint is based on the result of this function on the validation set
        device (torch.device): device to train on
        loaders (dict): 'train' and 'val' loaders
        checkpoint_dir (string): dir for saving checkpoints and tensorboard logs
        max_num_epochs (int): maximum number of epochs
        max_num_iterations (int): maximum number of iterations
        validate_after_iters (int): validate after that many iterations
        log_after_iters (int): number of iterations before logging
        validate_iters (int): number of validation iterations, if None validate
            on the whole validation set
        eval_score_higher_is_better (bool): if True higher eval scores are considered better
        best_eval_score (float): best validation score so far (higher better)
        num_iterations (int): useful when loading the model from the checkpoint
        num_epoch (int): useful when loading the model from the checkpoint
        skip_train_validation (bool): if True eval_criterion is not evaluated on the training set (used mostly when
            evaluation is expensive)
    """

    def __init__(self, model, optimizer, lr_scheduler, loss_criterion,
                 eval_criterion, web_logger, device, loaders, checkpoint_dir, verbose_train_validation,
                 max_num_epochs, max_num_iterations,
                 acc_batchsize=1, store_after_val=100,
                 validate_after_iters=200, log_after_iters=100, num_of_img_per_val=1,
                 validate_iters=None, num_iterations=1, num_epoch=0,
                 eval_score_higher_is_better=True, skip_train_validation=False,
                 resume=None, pre_trained=None, **kwargs):
        self.web_logger = web_logger
        self.acc_batchsize = acc_batchsize
        self.verbose_train_validation = verbose_train_validation,
        self.model = model
        self.optimizer = optimizer
        self.scheduler = lr_scheduler
        self.loss_criterion = loss_criterion
        self.eval_criterion = eval_criterion
        self.store_after_val = store_after_val
        self.device = device
        self.loaders = loaders
        self.checkpoint_dir = checkpoint_dir
        self.num_of_img_per_val = num_of_img_per_val
        self.max_num_epochs = max_num_epochs
        self.max_num_iterations = max_num_iterations
        self.validate_after_iters = validate_after_iters
        self.log_after_iters = log_after_iters
        self.validate_iters = validate_iters
        self.eval_score_higher_is_better = eval_score_higher_is_better

        logger.info(model)
        logger.info(f'eval_score_higher_is_better: {eval_score_higher_is_better}')

        # initialize the best_eval_score
        if eval_score_higher_is_better:
            self.best_eval_score = float('-inf')
        else:
            self.best_eval_score = float('+inf')

        self.num_iterations = num_iterations
        self.num_epochs = num_epoch
        self.skip_train_validation = skip_train_validation

        if resume is not None:
            logger.info(f"Loading checkpoint '{resume}'...")
            state = utils.load_checkpoint(resume, self.model, self.optimizer)
            logger.info(
                f"Checkpoint loaded from '{resume}'. Epoch: {state['num_epochs']}.  Iteration: {state['num_iterations']}. "
                f"Best val score: {state['best_eval_score']}."
            )
            self.best_eval_score = state['best_eval_score']
            self.num_iterations = state['num_iterations']
            self.num_epochs = state['num_epochs']
            self.checkpoint_dir = os.path.split(resume)[0]
        elif pre_trained is not None:
            logger.info(f"Logging pre-trained model from '{pre_trained}'...")
            utils.load_checkpoint(pre_trained, self.model, None)
            if 'checkpoint_dir' not in kwargs:
                self.checkpoint_dir = os.path.split(pre_trained)[0]

    def fit(self):
        try:
            for _ in range(self.num_epochs, self.max_num_epochs):
                # train for one epoch
                should_terminate = self.train()

                if should_terminate:
                    logger.info('Stopping criterion is satisfied. Finishing training')

                    return

                self.num_epochs += 1
            logger.info(f"Reached maximum number of epochs: {self.max_num_epochs}. Finishing training...")
        except KeyboardInterrupt:
            logger.info(f"Exit training by pressing CTRL+C ...")
            self.run_validation_step(store_model=True)

    def eval(self, output, target):
        if self.model.final_activation is not None:
            output = self.model.final_activation(output)
        return self.eval_criterion(output, target)

    def train(self):
        """Trains the model for 1 epoch.

        Returns:
            True if the training should be terminated immediately, False otherwise
        """
        self.train_losses = utils.RunningAverage()
        self.eval_score = utils.EvalScoreTracker()

        # sets the model in training mode
        self.model.train()
        self.optimizer.zero_grad()

        for t in self.loaders['train']:
            logger.info(f'Training iteration [{self.num_iterations}/{self.max_num_iterations}]. '
                        f'Epoch [{self.num_epochs}/{self.max_num_epochs - 1}]')

            input, target, weight = self._split_training_batch(t)

            output, loss = self._forward_pass(input, target, weight)
            self.train_losses.update(loss.item(), self._batch_size(input))

            if self.verbose_train_validation:
                eval_score = self.eval(output, target)
                self.eval_score.update(eval_score, self._batch_size(input))

            # compute gradients and update parameters
            loss /= self.acc_batchsize

            loss.backward()

            if self.num_iterations % self.acc_batchsize == 0 or self.acc_batchsize == 1:
                self.optimizer.step()
                self.optimizer.zero_grad()

            if self.num_iterations % self.validate_after_iters == 0:
                self.run_validation_step(
                    store_model=self.num_iterations % (self.store_after_val * self.validate_after_iters) == 0)

            if self.num_iterations % self.log_after_iters == 0:
                # compute eval criterion
                if not self.skip_train_validation and not self.verbose_train_validation:
                    eval_score = self.eval(output, target)
                    self.eval_score.update(eval_score, self._batch_size(input))

                eval_score, eval_score_detailed = self.eval_score.avg
                if hasattr(self.eval_criterion, "compute_final"):
                    eval_score, eval_score_detailed = self.eval_criterion.compute_final(self.eval_score)

                # log stats, params and images
                logger.info(
                    f'Training stats. Loss: {self.train_losses.avg}. Evaluation score: {eval_score}, {self.eval_score}')
                self.web_logger.log_stats(self.train_losses.avg, eval_score, eval_score_detailed, self.num_iterations,
                                          "train")
                self.web_logger.log_params(self.num_iterations)
                self.web_logger.log_images(input, target, output, self.num_iterations, 'train_')
                self.web_logger.log_images_upload(self.num_iterations, 'train_')

            if self.should_stop():
                self.run_validation_step(store_model=True)
                return True
            # wandb only upload on increasing steps
            # logging _non_ as dummy for the next step
            if self.num_iterations % self.validate_after_iters == 0 or self.num_iterations % self.log_after_iters == 0:
                self.web_logger.log_non(self.num_iterations + 1)

            self.num_iterations += 1

        return False

    def run_validation_step(self, store_model):
        # set the model in eval mode
        self.model.eval()
        # evaluate on validation set
        eval_score, eval_loss = self.validate()
        # set the model back to training mode
        self.model.train()

        # adjust learning rate if necessary
        if isinstance(self.scheduler, ReduceLROnPlateau):
            self.scheduler.step(eval_score)
        else:
            self.scheduler.step()
        # log current learning rate
        self.web_logger.log_learning_rate(self.optimizer.param_groups[0]['lr'], step=self.num_iterations)
        # remember best validation metric
        is_best = self._is_best_eval_score(eval_score)

        # save checkpoint
        if store_model:
            self._save_checkpoint(is_best, train_loss=self.train_losses.avg, train_eval=self.train_losses.avg,
                                  val_loss=eval_loss, val_eval=eval_score)

    def should_stop(self):
        """
        Training will terminate if maximum number of iterations is exceeded or the learning rate drops below
        some predefined threshold (1e-6 in our case)
        """
        if self.max_num_iterations < self.num_iterations:
            logger.info(f'Maximum number of iterations {self.max_num_iterations} exceeded.')
            return True

        min_lr = 1e-6
        lr = self.optimizer.param_groups[0]['lr']
        if lr < min_lr:
            logger.info(f'Learning rate below the minimum {min_lr}.')
            return True

        return False

    def validate(self):
        logger.info('Validating...')

        val_losses = utils.RunningAverage()
        val_scores = utils.EvalScoreTracker()

        img_idx = int(len(self.loaders['val']) / self.num_of_img_per_val)
        num_logged_img = 0
        with torch.no_grad():
            for i, t in enumerate(self.loaders['val']):
                logger.info(f'Validation iteration {i}')

                input, target, weight = self._split_training_batch(t)

                output, loss = self._forward_pass(input, target, weight)
                val_losses.update(loss.item(), self._batch_size(input))

                if i % img_idx == 0 and num_logged_img < self.num_of_img_per_val:
                    self.web_logger.log_images(input, target, output, self.num_iterations, 'val_')
                    num_logged_img += 1

                eval_score = self.eval(output, target)
                val_scores.update(eval_score, self._batch_size(input))

                if self.validate_iters is not None and self.validate_iters <= i:
                    # stop validation
                    break

            eval_score, eval_score_detailed = val_scores.avg
            if hasattr(self.eval_criterion, "compute_final"):
                eval_score, eval_score_detailed = self.eval_criterion.compute_final(val_scores)

            self.web_logger.log_images_upload(self.num_iterations, 'val_')
            self.web_logger.log_stats(val_losses.avg, eval_score, eval_score_detailed, self.num_iterations, "val")
            logger.info(f'Validation finished. Loss: {val_losses.avg}. Evaluation score: {eval_score}, {val_scores}')
            return eval_score, val_losses.avg

    def _split_training_batch(self, t):
        def _move_to_device(input):
            if isinstance(input, tuple) or isinstance(input, list):
                return tuple([_move_to_device(x) for x in input])
            else:
                return input.to(self.device)

        t = _move_to_device(t)
        weight = None
        if len(t) == 2:
            input, target = t
        else:
            input, target, weight = t
        return input, target, weight

    def _forward_pass(self, input, target, weight=None):
        # forward pass
        output = self.model(input)

        # compute the loss
        if weight is None:
            loss = self.loss_criterion(output, target)
        else:
            loss = self.loss_criterion(output, target, weight)

        return output, loss

    def _is_best_eval_score(self, eval_score):
        if self.eval_score_higher_is_better:
            is_best = eval_score > self.best_eval_score
        else:
            is_best = eval_score < self.best_eval_score

        if is_best:
            logger.info(f'Saving new best evaluation metric: {eval_score}')
            self.best_eval_score = eval_score

        return is_best

    def _save_checkpoint(self, is_best, train_loss, train_eval, val_loss, val_eval):
        # remove `module` prefix from layer names when using `nn.DataParallel`
        # see: https://discuss.pytorch.org/t/solved-keyerror-unexpected-key-module-encoder-embedding-weight-in-state-dict/1686/20
        if isinstance(self.model, nn.DataParallel):
            state_dict = self.model.module.state_dict()
        else:
            state_dict = self.model.state_dict()

        last_file_path = os.path.join(self.checkpoint_dir, 'last_checkpoint.pytorch')
        logger.info(f"Saving checkpoint to '{last_file_path}'")

        utils.save_checkpoint({
            'num_epochs': self.num_epochs + 1,
            'num_iterations': self.num_iterations,
            'model_state_dict': state_dict,
            'best_eval_score': self.best_eval_score,
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, is_best, checkpoint_dir=self.checkpoint_dir)
        self.web_logger.log_model(is_best, {
            'num_epochs': self.num_epochs + 1,
            'num_iterations': self.num_iterations,
            'train_loss': train_loss,
            'train_eval': train_eval,
            'val_loss': val_loss,
            'val_eval': val_eval,
        })

    @staticmethod
    def _batch_size(input):
        if isinstance(input, list) or isinstance(input, tuple):
            return input[0].size(0)
        else:
            return input.size(0)
