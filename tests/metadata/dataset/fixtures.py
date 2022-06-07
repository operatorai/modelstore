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

# pylint: disable=redefined-outer-name
# pylint: disable=missing-function-docstring


@pytest.fixture
def np_2d_array():
    return np.random.rand(10, 50)


@pytest.fixture
def np_1d_array():
    return np.array([1, 2, 1, 2, 1])


@pytest.fixture
def pd_dataframe():
    rows = []
    for _ in range(10):
        rows.append({f"col_{j}": j for j in range(50)})
    return pd.DataFrame(rows)


@pytest.fixture
def pd_series():
    return pd.Series([1, 2, 1, 2, 1])
