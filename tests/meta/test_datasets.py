#    Copyright 2020 Neal Lathia
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
import pandas as pd
import pytest
from modelstore.meta import datasets

# pylint: disable=redefined-outer-name


@pytest.fixture
def np_array():
    return np.random.rand(10, 50)


@pytest.fixture
def np_labels():
    return np.array([1, 2, 1, 2, 1])


@pytest.fixture
def pd_dataframe():
    rows = []
    for _ in range(20):
        rows.append({f"col_{j}": j for j in range(50)})
    return pd.DataFrame(rows)


@pytest.fixture
def pd_series():
    return pd.Series([1, 2, 1, 2, 1])


def test_is_numpy_array(np_array, pd_dataframe, pd_series):
    assert datasets.is_numpy_array(np_array)
    assert not datasets.is_numpy_array(pd_dataframe)
    assert not datasets.is_numpy_array(pd_series)


def is_pandas_dataframe(np_array, pd_dataframe, pd_series):
    assert not datasets.is_pandas_dataframe(np_array)
    assert datasets.is_pandas_dataframe(pd_dataframe)
    assert not datasets.is_pandas_dataframe(pd_series)


def test_is_pandas_series(np_array, pd_dataframe, pd_series):
    assert not datasets.is_pandas_series(np_array)
    assert not datasets.is_pandas_series(pd_dataframe)
    assert datasets.is_pandas_series(pd_series)


def test_describe_np_training(np_array):
    exp = {"shape": [10, 50]}
    res = datasets.describe_dataset(np_array)
    assert exp == res


def test_describe_np_labels(np_labels):
    exp = {"shape": [5], "values": {1: 3, 2: 2}}
    res = datasets.describe_dataset(np_labels)
    assert exp == res


def test_describe_df_training(pd_dataframe):
    exp = {"shape": [20, 50]}
    res = datasets.describe_dataset(pd_dataframe)
    assert exp == res


def test_describe_df_labels(pd_series):
    exp = {"shape": [5], "values": {1: 3, 2: 2}}
    res = datasets.describe_dataset(pd_series)
    assert exp == res
