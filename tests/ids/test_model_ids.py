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

import uuid
import pytest

from modelstore.ids import model_ids

# pylint: disable=protected-access
# pylint: disable=missing-function-docstring


def test_new() -> str:
    model_id = model_ids.new()
    assert isinstance(model_id, str)
    assert len(model_id) == len(str(uuid.uuid4()))


@pytest.mark.parametrize(
    "model_id,is_valid",
    [
        ("a-model-id", True),
        ("a model id", False),
    ],
)
def test_validate_no_spaces(model_id: str, is_valid: bool):
    assert model_ids.validate(model_id) == is_valid


def test_validate_no_special_characters():
    for character in model_ids._RESERVED_CHARACTERS:
        model_id = f"an-invalid-{character}-model-id"
        assert not model_ids.validate(model_id)
