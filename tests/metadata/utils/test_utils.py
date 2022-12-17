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
import pytest
import numpy as np

from modelstore.metadata.utils.utils import (
    remove_nones,
    exclude_field,
    validate_json_serializable,
)

# pylint: disable=missing-function-docstring


def test_remove_nones():
    exp = {"a": "value-a"}
    res = remove_nones({"a": "value-a", "b": None})
    assert exp == res


@pytest.mark.parametrize(
    "value,should_exclude",
    [
        (None, True),
        ("", False),
        (1, False),
    ],
)
def test_exclude_field(value, should_exclude):
    assert exclude_field(value) == should_exclude


@pytest.mark.parametrize(
    "value,should_raise",
    [
        ({}, False),
        ({"key": 1}, False),
        ([], True),  # Not a dictionary
        ({"key": np.array([1, 2, 3])}, True),  # Not JSON serializable
    ],
)
def test_validate_json_serializable(value, should_raise):
    """Validates that `value` is a JSON serializable dictionary"""
    if should_raise:
        with pytest.raises(TypeError):
            validate_json_serializable("field-name", value)
    else:
        try:
            validate_json_serializable("field-name", value)
            # pylint: disable=broad-except
        except Exception as exc:
            pytest.fail(f"validate_json_serializable() raised: {exc}")
