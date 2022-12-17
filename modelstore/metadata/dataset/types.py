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
import numpy as np

try:
    # numpy is a required dependency for modelstore,
    # but pandas is not
    import pandas as pd

    PANDAS_EXISTS = True
except ImportError:
    PANDAS_EXISTS = False


def is_numpy_array(values) -> bool:
    """Whether values is a numpy array"""
    return isinstance(values, np.ndarray)


def is_pandas_dataframe(values) -> bool:
    """Whether values is a pandas data frame"""
    if PANDAS_EXISTS:
        return isinstance(values, pd.DataFrame)
    return False


def is_pandas_series(values) -> bool:
    """Whether values is a pandas series"""
    if PANDAS_EXISTS:
        return isinstance(values, pd.Series)
    return False
