from typing import Optional

from modelstore.utils.log import logger

try:
    import numpy as np

    NUMPY_EXISTS = True
except ImportError:
    NUMPY_EXISTS = False

try:
    import pandas as pd

    PANDAS_EXISTS = True
except ImportError:
    PANDAS_EXISTS = False


def is_numpy_array(values):
    if NUMPY_EXISTS:
        return isinstance(values, np.ndarray)
    return False


def is_pandas_dataframe(values):
    if PANDAS_EXISTS:
        return isinstance(values, pd.DataFrame)
    return False


def is_pandas_series(values):
    if PANDAS_EXISTS:
        return isinstance(values, pd.Series)
    return False


def describe_training(dataset) -> Optional[dict]:
    """ Returns a description of a dataset"""
    if is_numpy_array(dataset):
        return {"shape": list(dataset.shape)}
    if is_pandas_dataframe(dataset):
        return {"shape": list(dataset.shape)}
    logger.debug(f"Trying to describe unknown type: {type(dataset)}")
    return None


def describe_labels(labels) -> Optional[dict]:
    if is_numpy_array(labels):
        unique, counts = np.unique(labels, return_counts=True)
        return {
            "shape": list(labels.shape),
            "values": dict(zip(unique, counts)),
        }
    if is_pandas_series(labels):
        return {
            "shape": list(labels.shape),
            "values": labels.value_counts().to_dict(),
        }
    logger.debug(f"Trying to describe unknown type: {type(labels)}")
    return None
