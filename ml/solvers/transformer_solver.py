import gc
import logging
import os
import sys
import time
import torch
from tqdm import tqdm

from data.datasets import get_dataloader
from ml.models import get_model
from ml.modules.losses import get_loss
from ml.optimizers import get_optimizer, get_lr_policy, get_lr_policy_parameter
from ml.solvers.base_solver import Solver
from utils.device import DEVICE, put_minibatch_to_device
from utils.get_model_inputs import get_model_inputs
from utils.iohandler import IOHandler


class TransformerSolver(Solver):
    def __init__(self, config, args):
        """
        Solver parent function to control the experiments.
        It contains everything for an experiment to run.

        :param config: config namespace containing the experiment configuration
        :param args: arguments of the training
        """
        super(TransformerSolver, self).__init__(config, args)

        # deep learn specific stuff
        self.init_epochs()
        if DEVICE == torch.device('cuda'): self.model.cuda()
        self.init_loss()
        self.init_optimizer()
        self.init_lr_policy()
        self.iohandler = IOHandler(args, self)
        # self.load_model()  # todo

    def init_epochs(self):
        """
        Initialize the epoch number initialization(s), (can be overwritten during checkpoint load).
        """
        logging.info("Initializing the epoch number.")
        self.epoch = 0
        self.epochs = self.config.env.epochs

    def init_loss(self):
        """
        Initialize the loss according to the config.
        """
        if self.phase == 'train':
            logging.info("Initializing the loss.")
            self.loss = get_loss(self.config.loss)

    def init_optimizer(self):
        """
        Initialize the optimizer according to the config, (can be overwritten during checkpoint load).
        """
        if self.phase == 'train':
            logging.info("Initializing the optimizer.")
            self.optimizer = get_optimizer(self.config.optimizer, self.model.parameters())

    def init_lr_policy(self):
        """
        Initialize the learning rate policy, (can be overwritten during checkpoint load).
        """
        if self.phase == 'train':
            logging.info("Initializing lr policy.")
            self.lr_policy = get_lr_policy(self.config.lr_policy, optimizer=self.optimizer)

    def train(self):
        """
        Training all the epochs with validation after every epoch.
        Save the model if it has better performance than the previous ones.
        """
        for self.epoch in range(self.epoch, self.epochs):
            logging.info(f"Start training epoch: {self.epoch}/{self.epochs}")
            self.current_mode = 'train'
            self.run_epoch()

            logging.info(f"Start evaluating epoch: {self.epoch}/{self.epochs}")
            self.eval()

            self.lr_policy.step(*get_lr_policy_parameter(self))
            self.iohandler.save_best_checkpoint()
            self.train_loader.dataset.sample_classes()

    def eval(self):
        """
        Evaluate the model and save the predictions to a csv file during testing inference.
        """
        self.current_mode = 'val'
        self.run_epoch()
        self.accuracy = self.iohandler.calculate_metric()
        print(self.accuracy)
        self.save_acc()
        # if self.args.save_preds: self.iohandler.save_results_csv()  # todo

    def before_epoch(self):
        """
        Before every epoch set the model and the iohandler to the right mode (train or eval)
        and select the corresponding loader.
        """
        self.iohandler.reset_results()
        if self.current_mode == 'train':
            self.model.train()
            self.loader = self.train_loader
            self.iohandler.train()
        elif self.current_mode == 'val':
            self.model.eval()
            self.loader = self.val_loader
            self.iohandler.val()
        else:
            raise ValueError(f'Wrong current mode: {self.current_mode}. It should be `train` or `val`.')
        torch.cuda.empty_cache()

    def run_epoch(self):
        """
        Run a full epoch according to the current self.current_mode (train or val).
        """
        self.before_epoch()

        # set loading bar
        bar_format = '{desc}|{bar:10}|[{elapsed}<{remaining},{rate_fmt}]'
        with tqdm(range(len(self.loader)), file=sys.stdout, bar_format=bar_format, position=0, leave=True) as pbar:
            preproc_t_start = time.time()
            for idx, minibatch in enumerate(self.loader):
                minibatch = put_minibatch_to_device(minibatch)
                preproc_time = time.time() - preproc_t_start

                # train
                train_t_start = time.time()
                output, loss = self.step(minibatch)
                train_time = time.time() - train_t_start

                # save results for evaluation at the end of the epoch and calculate the running metrics
                self.iohandler.append_data(minibatch, output)
                self.iohandler.update_bar_description(pbar, idx, preproc_time, train_time, loss)

                pbar.update(1)
                preproc_t_start = time.time()

        self.after_epoch()

    def step(self, minibatch):
        """
        Make one iteration step: either a train (pred+train) or a val step (pred only).

        :param minibatch: minibatch containing the input image and the labels (labels only during `train`).
        :return: output, loss
        """
        # prediction
        if self.current_mode == 'train':
            output = self.model(**get_model_inputs(self.model, minibatch))

            # training step
            self.optimizer.zero_grad()
            loss = self.loss(output, minibatch['labels'].unsqueeze(-1))
            loss.backward()
            self.optimizer.step()

            output = {'output': output}
        else:
            with torch.no_grad():
                if self.current_mode == 'val':
                    model_inputs = get_model_inputs(self.model, minibatch)
                    less_output = self.model(**{key: value[:, 0] for key, value in model_inputs.items()})
                    more_output = self.model(**{key: value[:, 1] for key, value in model_inputs.items()})
                    output = {'less_output': less_output, 'more_output': more_output}
                elif self.current_mode == 'test':
                    output = self.model(**get_model_inputs(self.model, minibatch))
                    output = {'output': output}
                else:
                    raise ValueError(f'Wrong current mode: {self.current_mode}')
            loss = 0

        return output, loss

    def after_epoch(self):
        """
        After every epoch collect some garbage and evaluate the current metric.
        """
        gc.collect()
        torch.cuda.empty_cache()
        print()
