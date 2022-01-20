import logging
import os
import pandas as pd
from torch.utils.data import Dataset
import numpy as np

from data.utils.clean_text import clean
from ml.tokenizers import get_tokenizer
from ml.vectorizers import get_vectorizer
from tqdm.auto import tqdm
tqdm.pandas()


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

    def __getitem__(self, index):
        if self.split == 'train':
            return {"id": self.id[index],
                    'ids': self.inputs[index],
                    "mask": self.attention_masks[index],
                    'labels': self.outputs}
        elif self.split == 'val' or self.split == 'test':
            return {"id": self.id[index],
                    'ids': self.inputs[index],
                    "mask": self.attention_masks[index]}
        else:
            raise ValueError(f'Wrong split: {self.split}')

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
            self.id = self.df['id'].to_numpy()
            self.inputs = self.df['comment_text']
            self.make_outputs()
            self.outputs = self.df['y'].to_numpy()
        elif self.split == 'val':  # todo: change it
            self.id = [None] * len(self.df)
            self.inputs = self.df['less_toxic'], self.df['more_toxic']
            self.outputs = [None] * len(self.df)
        elif self.split == 'test':
            self.id = self.df['comment_id'].to_numpy()
            self.inputs = self.df['text']
            self.outputs = [None] * len(self.df)
        else:
            raise ValueError(f'Wrong split: {self.split}')

        self.clean_text()

    def clean_text(self):
        if self.config.clean_text:
            if self.split == 'train' or self.split == 'test':
                self.inputs = self.inputs.progress_apply(clean)
            elif self.split == 'val':  # todo: change it
                self.inputs = self.inputs[0].progress_apply(clean), \
                              self.inputs[0].progress_apply(clean)
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
            tokenized = JigsawDataloader.tokenizer(list(self.inputs),
                                                   **self.config.tokenizer.tokenize_params.dict())
            tokenized_comments = np.array(tokenized['input_ids'])
            self.attention_masks = np.array(tokenized['attention_mask'])
            self.inputs = JigsawDataloader.vectorizer.fit_transform(tokenized_comments)
        elif self.split == 'val':
            tokenized_less = JigsawDataloader.tokenizer(list(self.inputs[0]),
                                                        **self.config.tokenizer.tokenize_params.dict())
            tokenized_comments_less = np.array(tokenized_less['input_ids'])
            self.attention_masks = np.array(tokenized_less['attention_mask'])
            tokenized_more = JigsawDataloader.tokenizer(list(self.inputs[1]),
                                                        **self.config.tokenizer.tokenize_params.dict())
            tokenized_comments_more = np.array(tokenized_more['input_ids'])
            self.attention_masks = np.array(tokenized_more['attention_mask'])
            self.inputs = JigsawDataloader.vectorizer.transform(tokenized_comments_less), \
                          JigsawDataloader.vectorizer.transform(tokenized_comments_more)
        elif self.split == 'test':
            tokenized = JigsawDataloader.tokenizer(list(self.inputs),
                                                   **self.config.tokenizer.tokenize_params.dict())
            tokenized_comments = np.array(tokenized['input_ids'])
            self.attention_masks = np.array(tokenized['attention_mask'])
            self.inputs = JigsawDataloader.vectorizer.transform(tokenized_comments)
        else:
            raise ValueError(f'Wrong split: {self.split}')
