from sklearn.feature_extraction import text


def get_vectorizer(config):
    """
    Select the vectorizer according to the config name and its parameters

    :param config: config containing the name as config.name and the parameters as config.params
    :return: tokenizer
    """

    if hasattr(text, config.name):
        function = getattr(text, config.name)
        return function(analyzer='word',  # todo place it to somewhere else
                        tokenizer=lambda x: x,
                        preprocessor=lambda x: x,
                        token_pattern=None)
    elif config.name is None:
        return lambda x: x
    else:
        raise ValueError(f'Wrong model name in model configs: {config.name}')
