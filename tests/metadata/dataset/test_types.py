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
from modelstore.metadata.dataset.types import (
    is_numpy_array,
    is_pandas_dataframe,
    is_pandas_series,
)

# pylint: disable=unused-import
from tests.metadata.dataset.fixtures import (
    np_1d_array,
    np_2d_array,
    pd_dataframe,
    pd_series,
)

# pylint: disable=redefined-outer-name
# pylint: disable=missing-function-docstring


def test_is_numpy_array(np_1d_array, np_2d_array, pd_dataframe, pd_series):
    assert is_numpy_array(np_1d_array)
    assert is_numpy_array(np_2d_array)
    assert not is_numpy_array(pd_dataframe)
    assert not is_numpy_array(pd_series)


def is_pandas_dataframe(np_1d_array, np_2d_array, pd_dataframe, pd_series):
    assert not is_pandas_dataframe(np_1d_array)
    assert not is_pandas_dataframe(np_2d_array)
    assert is_pandas_dataframe(pd_dataframe)
    assert not is_pandas_dataframe(pd_series)


def test_is_pandas_series(np_1d_array, np_2d_array, pd_dataframe, pd_series):
    assert not is_pandas_series(np_1d_array)
    assert not is_pandas_series(np_2d_array)
    assert not is_pandas_series(pd_dataframe)
    assert is_pandas_series(pd_series)
