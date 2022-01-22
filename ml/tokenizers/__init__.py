from tokenizers import models, normalizers, pre_tokenizers, trainers, Tokenizer
from transformers import PreTrainedTokenizerFast, AutoTokenizer


def get_tokenizer(config, train_dataset):
    """
    Select the tokenizer according to the config name and its parameters

    :param config: config containing the name as config.name and the parameters as config.params
    :param train_dataset: train dataset containing the training text
    :return: tokenizer
    """
    if config.pretrained:
        return AutoTokenizer.from_pretrained(config.name)
    elif hasattr(models, config.name):
        function = getattr(models, config.name)
        raw_tokenizer = Tokenizer(function(**config.params.dict()))
        raw_tokenizer.normalizer = get_normalizer(config.normalizer)
        raw_tokenizer.pre_tokenizer = get_pre_tokenizer(config.pre_tokenizer)
        trainer = get_trainer(config.trainer)
        raw_tokenizer.train_from_iterator(train_dataset, trainer)
        return PreTrainedTokenizerFast(tokenizer_object=raw_tokenizer,
                                       unk_token="[UNK]",
                                       pad_token="[PAD]",
                                       cls_token="[CLS]",
                                       sep_token="[SEP]",
                                       mask_token="[MASK]")
    elif config.name is None:
        return lambda x: x
    else:
        raise ValueError(f'Wrong model name in model configs: {config.name}')


def get_normalizer(config):
    if hasattr(normalizers, config.name):
        function = getattr(normalizers, config.name)
        return function(**config.params.dict())


def get_pre_tokenizer(config):
    if hasattr(pre_tokenizers, config.name):
        function = getattr(pre_tokenizers, config.name)
        return function(**config.params.dict())


def get_trainer(config):
    if hasattr(trainers, config.name):
        function = getattr(trainers, config.name)
        return function(**config.params.dict())
