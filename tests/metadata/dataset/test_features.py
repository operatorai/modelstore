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
from modelstore.metadata.dataset.features import Features

# pylint: disable=unused-import
from tests.metadata.dataset.fixtures import (
    np_2d_array,
    np_1d_array,
    pd_dataframe,
)

# pylint: disable=redefined-outer-name
# pylint: disable=missing-function-docstring


def test_describe_nothing():
    res = Features.generate()
    assert res is None


def test_describe_numpy_2d(np_2d_array):
    exp = Features(shape=[10, 50])
    res = Features.generate(np_2d_array)
    assert exp == res


def test_describe_numpy_1d(np_1d_array):
    exp = Features(shape=[5])
    res = Features.generate(np_1d_array)
    assert exp == res


def test_describe_dataframe(pd_dataframe):
    exp = Features(shape=[10, 50])
    res = Features.generate(pd_dataframe)
    assert exp == res
