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

        # undersample the overpresented, and oversample the underpresented classes
        if self.split == 'train':
            self.old_id, self.old_inputs, self.old_attention_masks, self.old_outputs = \
                self.id, self.inputs, self.attention_masks, self.outputs
            self.sample_classes()

    @property
    def data(self):
        return self.inputs, self.outputs

    def __getitem__(self, index):
        if self.split == 'train':
            return {"id": self.id[index],
                    'ids': self.inputs[index],
                    "mask": self.attention_masks[index],
                    'labels': self.outputs[index]}
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
            filepath = os.path.join(self.config.dataset_path, 'comments_to_score.csv')
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
            self.outputs = self.df['y'].to_numpy().astype(np.float32)
        elif self.split == 'val':
            self.id = [0] * len(self.df)
            self.inputs = self.df['less_toxic'], self.df['more_toxic']
        elif self.split == 'test':
            self.id = self.df['comment_id'].to_numpy()
            self.inputs = self.df['text']
        else:
            raise ValueError(f'Wrong split: {self.split}')

        self.clean_text()

    def clean_text(self):
        logging.info('Cleaning data')
        if self.config.clean_text:
            if self.split == 'train' or self.split == 'test':
                self.inputs = self.inputs.progress_apply(clean)
            elif self.split == 'val':  # todo: change it
                self.inputs = self.inputs[0].progress_apply(clean), \
                              self.inputs[1].progress_apply(clean)
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
            tokenized_comments_less = JigsawDataloader.vectorizer.transform(tokenized_comments_less)
            attention_masks_less = np.array(tokenized_less['attention_mask'])
            tokenized_more = JigsawDataloader.tokenizer(list(self.inputs[1]),
                                                        **self.config.tokenizer.tokenize_params.dict())
            tokenized_comments_more = np.array(tokenized_more['input_ids'])
            tokenized_comments_more = JigsawDataloader.vectorizer.transform(tokenized_comments_more)
            attention_masks_more = np.array(tokenized_more['attention_mask'])
            self.attention_masks = np.array(list(zip(attention_masks_less, attention_masks_more)))
            self.inputs = np.array(list(zip(tokenized_comments_less, tokenized_comments_more)))
        elif self.split == 'test':
            tokenized = JigsawDataloader.tokenizer(list(self.inputs),
                                                   **self.config.tokenizer.tokenize_params.dict())
            tokenized_comments = np.array(tokenized['input_ids'])
            self.attention_masks = np.array(tokenized['attention_mask'])
            self.inputs = JigsawDataloader.vectorizer.transform(tokenized_comments)
        else:
            raise ValueError(f'Wrong split: {self.split}')

    def sample_classes(self):
        """
        Make the over or undersampling. This should be rerun after every epoch for
        new samples to use all the samples from the dataset.
        """
        logging.info('Resampling dataset to have equal positives and negatives.')

        if not self.config.balanced_classes and self.split == 'train':
            return

        istoxic = (self.df.loc[:, 'toxic':'identity_hate'].sum(1) > 0).astype(int)

        id, inputs, attention_masks, outputs = [], [], [], []

        for class_number in range(2):
            class_mask = istoxic == class_number

            masked_id = self.old_id[class_mask]
            masked_inputs = self.old_inputs[class_mask]
            masked_attention_masks = self.old_attention_masks[class_mask]
            masked_outputs = self.old_outputs[class_mask]

            chosen_indices = np.random.choice(np.arange(len(masked_inputs)), size=25000)

            id.extend(masked_id[chosen_indices])
            inputs.extend(masked_inputs[chosen_indices])
            attention_masks.extend(masked_attention_masks[chosen_indices])
            outputs.extend(masked_outputs[chosen_indices])

        self.id, self.inputs, self.attention_masks, self.outputs = id, inputs, attention_masks, outputs