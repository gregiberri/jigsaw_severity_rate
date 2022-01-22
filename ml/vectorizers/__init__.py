from sklearn.feature_extraction import text


class EmptyVectorizer:
    def fit_transform(self, input):
        return input

    def transform(self, input):
        return input


def get_vectorizer(config):
    """
    Select the vectorizer according to the config name and its parameters

    :param config: config containing the name as config.name and the parameters as config.params
    :return: tokenizer
    """

    if config is None:
        return EmptyVectorizer()
    elif hasattr(text, config.name):
        function = getattr(text, config.name)
        return function(analyzer='word',  # todo dynamic with config file
                        tokenizer=lambda x: x,
                        preprocessor=lambda x: x,
                        token_pattern=None)
    else:
        raise ValueError(f'Wrong model name in model configs: {config.name}')
