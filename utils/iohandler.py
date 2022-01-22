import logging
import os

import torch
from sklearn.metrics import accuracy_score

from utils.device import DEVICE


class IOHandler:
    def __init__(self, args, solver):
        self.args = args
        self.phase = args.mode
        self.current_phase = self.phase
        self.solver = solver
        self.config = self.solver.config

        self.init_results_dir()
        self.metric = {}
        self.load_checkpoint()
        self.reset_results()

    def train(self):
        self.current_phase = 'train'

    def val(self):
        self.current_phase = 'val'

    def test(self):
        self.current_phase = 'test'

    def reset_results(self):
        """
        Reset the results to be empty before starting an epoch.
        """
        self.results = {}

    def get_max_metric(self):
        """
        Get the validation goal_metrics of the best model.
        """
        return {key: max(value) for key, value in self.metric.items()}

    def init_results_dir(self):
        """
        Making results dir.
        """
        logging.info("Making result dir.")
        result_name = os.path.join(self.config.id, self.args.id_tag) if self.args.id_tag else self.config.id
        self.result_dir = os.path.join(self.config.env.result_dir, result_name)
        if not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir)
        self.solver.config.save(os.path.join(self.result_dir, 'config.yml'))

        logging.info(f"Results dir is made. Results will be saved at: {self.result_dir}")

    def append_data(self, minibatch, output):
        if self.current_phase != 'train':
            for key, value in output.items():
                if not key in self.results:
                    self.results[key] = torch.tensor([], device=DEVICE)
                self.results[key] = torch.cat([self.results[key], value], dim=0)

    def calculate_metric(self):
        accuracy = (self.results['less_output'] < self.results['more_output']).cpu().numpy()
        accuracy = accuracy.mean()
        self.append_metric({'accuracy': accuracy})

        return accuracy

    def append_metric(self, metric):
        for key, value in metric.items():
            if not key in self.metric:
                self.metric[key] = []
            self.metric[key].append(float(value))

    def update_bar_description(self, pbar, idx, preproc_time, train_time, loss):
        """
        Update the current log bar with the latest result.

        :param pbar: pbar object
        :param idx: iteration number in the epoch
        :param preproc_time: time spent with preprocessing
        :param train_time: time spent with training
        :param loss: loss value
        """
        if self.phase == 'train':
            print_str = f'[{self.solver.current_mode}] epoch {self.solver.epoch}/{self.solver.epochs} ' \
                        + f'iter {idx + 1}/{len(self.solver.loader)}:' \
                        + f'lr:{self.solver.optimizer.param_groups[0]["lr"]:.5f}|' \
                        + f'loss: {loss:.3f}|' \
                        + f'|t_prep: {preproc_time:.3f}s|' \
                        + f't_train: {train_time:.3f}s'
        else:
            print_str = f'[{self.solver.current_mode}] ' \
                        + f'iter {idx + 1}/{len(self.solver.loader)}:' \
                        + f'|t_prep: {preproc_time:.3f}s|' \
                        + f't_train: {train_time:.3f}s'
        pbar.set_description(print_str, refresh=False)

    def save_best_checkpoint(self):
        """
        Save the model if the last epoch result is the best.
        """
        if not max(self.metric['accuracy']) == self.metric['accuracy'][-1]:
            return

        path = os.path.join(self.result_dir, 'model_best.pth.tar')

        state_dict = {'epoch': self.solver.epoch,
                      'optimizer': self.solver.optimizer.state_dict(),
                      'lr_policy': self.solver.lr_policy.state_dict(),
                      'config': self.solver.config,
                      'model': self.solver.model.state_dict()}

        torch.save(state_dict, path)
        del state_dict
        logging.info(f"Saved checkpoint to file {path}\n")

    def load_checkpoint(self):
        """
        If a saved model in the result folder exists load the model
        and the hyperparameters from a trained model checkpoint.
        """
        path = os.path.join(self.result_dir, 'model_best.pth.tar')
        if not os.path.exists(path):
            assert self.phase == 'train', f'No model file found to load at: {path}'
            return

        logging.info(f"Loading the checkpoint from: {path}")
        continue_state_object = torch.load(path, map_location=torch.device("cpu"))

        # load the things from the checkpoint
        if self.phase == 'train':
            self.solver.optimizer.load_state_dict(continue_state_object['optimizer'])
            self.solver.lr_policy.load_state_dict(continue_state_object['lr_policy'])

        self.solver.epoch = continue_state_object['epoch']
        # self.solver.config = continue_state_object['config']
        self.solver.model.load_state_dict(continue_state_object['model'])
        if DEVICE == torch.device('cuda'): self.solver.model.cuda()

        del continue_state_object
