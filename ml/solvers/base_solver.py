import csv
import os
from abc import ABC
import logging

from sklearn.feature_extraction.text import TfidfVectorizer

from data.datasets import get_dataloader
from ml.models import get_model
from ml.tokenizers import get_tokenizer


class Solver(ABC):
    def __init__(self, config, args):
        """
        Solver parent function to control the experiments.
        It contains everything for an experiment to run.

        :param config: config namespace containing the experiment configuration
        :param args: arguments of the training
        """
        self.args = args
        self.phase = args.mode
        self.config = config

        self.accuracy = 0

        # initialize the required elements for the ml problem
        self.init_results_dir()
        self.init_dataloaders()
        self.init_tokenizer()
        self.init_vectorizer()

        self.tokenize()

        self.init_model()
        # self.load_model()

    def init_results_dir(self):
        self.result_dir = os.path.join(self.config.env.result_dir, self.config.id, self.args.id_tag)
        if not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir)

    def init_tokenizer(self):
        """
        Initialize the tokenizer according to the config.
        """
        logging.info("Initializing the tokenizer.")
        self.tokenizer = get_tokenizer(self.config.tokenizer, self.train_loader.dataset.inputs)

    def init_vectorizer(self):
        """
        Initializer the vectorizer according to the config.
        """
        logging.info(f'Initializing the vectorizer.')
        self.vectorizer = TfidfVectorizer(analyzer='word',  # todo place it to somewhere else
                                          tokenizer=lambda x: x,
                                          preprocessor=lambda x: x,
                                          token_pattern=None)

    def tokenize(self):
        logging.info('Tokenizing.')
        tokenized_comments = self.tokenizer(list(self.train_loader.dataset.inputs))['input_ids']
        self.train_loader.dataset.inputs = self.vectorizer.fit_transform(tokenized_comments)

        tokenized_comments_less = self.tokenizer(list(self.val_loader.dataset.inputs[0]))['input_ids']
        tokenized_comments_more = self.tokenizer(list(self.val_loader.dataset.inputs[1]))['input_ids']
        self.val_loader.dataset.inputs = self.vectorizer.transform(tokenized_comments_less), \
                                         self.vectorizer.transform(tokenized_comments_more)

        # tokenized_comments = self.tokenizer(list(self.test_loader.dataset.inputs))['input_ids']
        # self.train_loader.dataset.inputs = self.vectorizer.fit_transform(tokenized_comments)

    def init_model(self):
        """
        Initialize the model according to the config and put it on the gpu if available,
        (weights can be overwritten during checkpoint load).
        """
        logging.info("Initializing the model.")
        self.model = get_model(self.config.model)

    def load_model(self):
        raise NotImplementedError

    def init_dataloaders(self):
        """
        Dataloader initialization(s) for train, val dataset according to the config.
        """
        logging.info("Initializing dataloaders.")
        if self.phase == 'train':
            self.train_loader = get_dataloader(self.config.data, 'train')
            self.val_loader = get_dataloader(self.config.data, 'val')
        elif self.phase == 'val':
            self.val_loader = get_dataloader(self.config.data, 'val')
        elif self.phase == 'test':
            self.test_loader = get_dataloader(self.config.data, 'test')
        else:
            raise ValueError(f'Wrong mode argument: {self.phase}. It should be `train`, `val` or `test`.')

    def run(self):
        """
        Run the experiment.
        :return: the best goal metrics (as stated in config.metrics.goal_metric).
        """
        logging.info("Starting experiment.")
        if self.phase == 'train':
            self.train()
        elif self.phase in ['val', 'test']:
            self.eval()
        else:
            raise ValueError(f'Wrong phase: {self.phase}. It should be `train`, `val`.')

        return self.accuracy

    def save_preds(self, preds):
        with open(os.path.join(self.result_dir, 'preds.csv'), 'w') as myfile:
            wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
            wr.writerow(['index', 'pred'])
            indices = self.test_loader.dataset.id.tolist()
            wr.writerows(zip(indices, preds))

    def save_acc(self):
        with open(os.path.join(self.result_dir, 'score.txt'), 'w') as myfile:
            myfile.write(f'lb_score: {self.accuracy}')

    def train(self):
        raise NotImplementedError

    def eval(self):
        raise NotImplementedError

    def test(self):
        raise NotImplementedError
