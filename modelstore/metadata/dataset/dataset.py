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
from typing import List

from dataclasses import dataclass
from dataclasses_json import dataclass_json

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


@dataclass_json
@dataclass
class Dataset:

    """ Dataset contains fields that are captured about
    the training dataset when the model is saved """

    shape: List[int]
    values: dict

    @classmethod
    def generate(cls, x, y) -> "Dataset":
        """ Returns summary stats about a dataset """
        pass

    @classmethod
    def describe(cls, dataset) -> "Dataset":
        """Returns summary stats about a dataset"""
        if is_numpy_array(dataset):
            if dataset.ndim == 1:
                # Array has one dimension (e.g., labels); return its
                # its shape and value counts
                unique, counts = np.unique(dataset, return_counts=True)
                return Dataset(
                    shape=list(dataset.shape),
                    values=dict(zip(unique, counts))
                )
            # Array is multi-dimensional, only return its shape
            return Dataset(
                shape=list(dataset.shape),
                values=None,
            )
        if is_pandas_dataframe(dataset):
            # Data frame can have multiple dimensions; only
            # return its shape
            return Dataset(
                shape=list(dataset.shape),
                values=None,
            )
        if is_pandas_series(dataset):
            # Data series has one dimension (e.g., labels); return
            # its shape and value counts
            return Dataset(
                shape=list(dataset.shape),
                values=dataset.value_counts().to_dict(),
            )
        logger.debug("Trying to describe unknown type: %s", type(dataset))
        return None
