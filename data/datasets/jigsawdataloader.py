import logging
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from torch.utils.data import Dataset

from ml.tokenizers import get_tokenizer
from ml.vectorizers import get_vectorizer


class JigsawDataloader(Dataset):
    """
    Dataloader to load the traffic signs.
    """
    tokenizer = None
    vectorizer = None

    def __init__(self, config, split):
        self.config = config
        self.split = split

        self.df = self.load_data()
        self.make_inputs_outputs()
        self.init_tokenizer()
        self.init_vectorizer()
        self.tokenize()

    @property
    def data(self):
        return self.inputs, self.outputs

    def __len__(self):
        return len(self.id)

    def load_data(self):
        """
        Load data from the corresponding csv file.

        :return: data pandas dataframe
        """
        logging.info(f'Loading {self.split} data')
        if self.split == 'val':
            filepath = os.path.join(self.config.dataset_path, 'validation_data.csv')
        elif self.split == 'train':
            filepath = os.path.join(self.config.dataset_path, 'train.csv')
        elif self.split == 'test':
            filepath = os.path.join(self.config.dataset_path, self.config.test_filename)
        else:
            raise ValueError(f'Wrong split: {self.split}.')

        assert os.path.exists(filepath), f'The required data file path ({filepath}) ' \
                                         f'does not exists'
        return pd.read_csv(filepath)

    def make_outputs(self):
        """
        Make outputs from the data.
        """
        for category, weight in self.config.category_weights.dict().items():
            self.df[category] = self.df[category] * weight

        self.df['y'] = self.df.loc[:, 'toxic':'identity_hate'].mean(axis=1)

    def make_inputs_outputs(self):
        """
        Make the inputs and the outputs according to the current split.
        """
        if self.split == 'train':
            self.id = self.df['id']
            self.inputs = self.df['comment_text']
            self.make_outputs()
            self.outputs = self.df['y']
        elif self.split == 'val':  # todo: change it
            self.id = None
            self.inputs = self.df['less_toxic'], self.df['more_toxic']
            self.outputs = None
        elif self.split == 'test':
            self.id = self.df['comment_id']
            self.inputs = self.df['text']
            self.outputs = None
        else:
            raise ValueError(f'Wrong split: {self.split}')

    def init_tokenizer(self):
        """
        Initialize the tokenizer according to the config.
        """
        logging.info("Initializing the tokenizer.")
        if self.split == 'train':
            JigsawDataloader.tokenizer = get_tokenizer(self.config.tokenizer, self.inputs)
        elif JigsawDataloader.tokenizer is None:
            raise NotImplementedError()

    def init_vectorizer(self):
        """
        Initializer the vectorizer according to the config.
        """
        logging.info(f'Initializing the vectorizer.')
        if self.split == 'train':
            JigsawDataloader.vectorizer = get_vectorizer(self.config.vectorizer)
        elif JigsawDataloader.vectorizer is None:
            raise NotImplementedError()

    def tokenize(self):
        if self.split == 'train':
            tokenized_comments = JigsawDataloader.tokenizer(list(self.inputs))['input_ids']
            self.inputs = JigsawDataloader.vectorizer.fit_transform(tokenized_comments)
        elif self.split == 'val':
            tokenized_comments_less = JigsawDataloader.tokenizer(list(self.inputs[0]))['input_ids']
            tokenized_comments_more = JigsawDataloader.tokenizer(list(self.inputs[1]))['input_ids']
            self.inputs = JigsawDataloader.vectorizer.transform(tokenized_comments_less), \
                          JigsawDataloader.vectorizer.transform(tokenized_comments_more)
        elif self.split == 'test':
            tokenized_comments = JigsawDataloader.tokenizer(list(self.inputs))['input_ids']
            self.inputs = JigsawDataloader.vectorizer.fit_transform(tokenized_comments)
        else:
            raise ValueError(f'Wrong split: {self.split}')
