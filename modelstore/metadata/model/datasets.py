#    Copyright 2022 Neal Lathia
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
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


def is_numpy_array(values) -> bool:
    """ Whether values is a numpy array """
    if NUMPY_EXISTS:
        return isinstance(values, np.ndarray)
    return False


def is_pandas_dataframe(values) -> bool:
    """ Whether values is a pandas data frame"""
    if PANDAS_EXISTS:
        return isinstance(values, pd.DataFrame)
    return False


def is_pandas_series(values) -> bool:
    """ Whether values is a pandas series """
    if PANDAS_EXISTS:
        return isinstance(values, pd.Series)
    return False


def describe_dataset(dataset) -> Optional[dict]:
    """Returns summary stats about a dataset"""
    if is_numpy_array(dataset):
        if dataset.ndim == 1:
            # Array has one dimension (e.g., labels); return its
            # its shape and value counts
            unique, counts = np.unique(dataset, return_counts=True)
            return {
                "shape": list(dataset.shape),
                "values": dict(zip(unique, counts)),
            }
        # Array is multi-dimensional, only return its shape
        return {"shape": list(dataset.shape)}
    if is_pandas_dataframe(dataset):
        # Data frame can have multiple dimensions; only
        # return its shape
        return {"shape": list(dataset.shape)}
    if is_pandas_series(dataset):
        # Data series has one dimension (e.g., labels); return
        # its shape and value counts
        return {
            "shape": list(dataset.shape),
            "values": dataset.value_counts().to_dict(),
        }
    logger.debug("Trying to describe unknown type: %s", type(dataset))
    return None
