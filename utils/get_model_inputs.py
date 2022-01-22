import inspect


def get_model_inputs(model, inputs):
    model_inputs = inspect.signature(model.forward).parameters.keys()
    return {model_input: inputs[model_input] for model_input in model_inputs}
