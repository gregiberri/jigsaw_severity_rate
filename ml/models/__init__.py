import xgboost
from sklearn import linear_model, naive_bayes, ensemble, tree
from transformers import AutoModel

from ml.models.model_wrapper import JigsawModelWrapper


def get_model(config):
    """
    Select the model according to the model config name and its parameters

    :param config: config containing the model name as config.name and the parameters as config.params
    :return: model
    """
    if config.pretrained:
        model = AutoModel.from_pretrained(config.name)
        return JigsawModelWrapper(model)

    elif hasattr(linear_model, config.name):
        function = getattr(linear_model, config.name)
        model = function(**config.params.dict())
        return model

    if hasattr(naive_bayes, config.name):
        function = getattr(naive_bayes, config.name)
        model = function(**config.params.dict())
        return model

    if hasattr(ensemble, config.name):
        function = getattr(ensemble, config.name)
        return function(**config.params.dict())

    if hasattr(tree, config.name):
        function = getattr(tree, config.name)
        return function(**config.params.dict())

    if hasattr(xgboost, config.name):
        config.params.max_depth = int(config.params.max_depth)
        config.params.n_estimators = int(config.params.n_estimators)
        function = getattr(xgboost, config.name)
        return function(**config.params.dict())
    else:
        raise ValueError(f'Wrong model name in model configs: {config.name}')




