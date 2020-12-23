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
    res = datasets.describe_training(np_array)
    assert exp == res


def test_describe_np_labels(np_labels):
    exp = {"shape": [5], "values": {1: 3, 2: 2}}
    res = datasets.describe_labels(np_labels)
    assert exp == res


def test_describe_df_training(pd_dataframe):
    exp = {"shape": [20, 50]}
    res = datasets.describe_training(pd_dataframe)
    assert exp == res


def test_describe_df_labels(pd_series):
    exp = {"shape": [5], "values": {1: 3, 2: 2}}
    res = datasets.describe_labels(pd_series)
    assert exp == res
