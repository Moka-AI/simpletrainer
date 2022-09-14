import importlib.util


def is_wandb_available():
    return importlib.util.find_spec('wandb') is not None


def is_transformers_available():
    return importlib.util.find_spec('transformers') is not None


def is_torchmetrics_available():
    return importlib.util.find_spec('torchmetrics') is not None


def is_mlflow_available():
    return importlib.util.find_spec('mlflow') is not None
