from ml.solvers.sklearn_solver import SKLearnSolver
from ml.solvers.transformer_solver import TransformerSolver


def get_solver(config, args):
    if config.env.solver == 'sklearn-solver':
        return SKLearnSolver(config, args)
    elif config.env.solver == 'transformer-solver':
        return TransformerSolver(config, args)

    else:
        raise ValueError(f'Wrong solver config: {config.env.solver}')
