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


def describe_training(dataset) -> Optional[dict]:
    """ Returns a description of a dataset"""
    if NUMPY_EXISTS:
        if isinstance(dataset, np.ndarray):
            return {"shape": list(dataset.shape)}
    if PANDAS_EXISTS:
        if isinstance(dataset, pd.DataFrame):
            return {"shape": list(dataset.shape)}
    logger.debug(f"Trying to describe unknown type: {type(dataset)}")
    return None


def describe_labels(labels) -> Optional[dict]:
    if NUMPY_EXISTS:
        if isinstance(labels, np.ndarray):
            unique, counts = np.unique(labels, return_counts=True)
            return {
                "shape": list(labels.shape),
                "values": dict(zip(unique, counts)),
            }
    if PANDAS_EXISTS:
        if isinstance(labels, pd.Series):
            return {
                "shape": list(labels.shape),
                "values": labels.value_counts().to_dict(),
            }
    logger.debug(f"Trying to describe unknown type: {type(labels)}")
    return None
