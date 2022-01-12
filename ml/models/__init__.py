from sklearn import linear_model, naive_bayes


def get_model(model_config):
    """
    Select the model according to the model config name and its parameters

    :param model_config: config containing the model name as config.name and the parameters as config.params
    :return: model
    """

    if hasattr(linear_model, model_config.name):
        function = getattr(linear_model, model_config.name)
        model = function(**model_config.params.dict())
        return model
    if hasattr(naive_bayes, model_config.name):
        function = getattr(naive_bayes, model_config.name)
        model = function(**model_config.params.dict())
        return model
    else:
        raise ValueError(f'Wrong model name in model configs: {model_config.name}')

