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

from modelstore.metadata import metadata

# pylint: disable=missing-function-docstring
# pylint: disable=redefined-outer-name


@pytest.fixture
def model_meta_data():
    return metadata.Model(
        domain="domain",
        model_id="model_id",
        model_type=metadata.ModelType.generate("library", "class-name"),
        parameters=None,
        data=None,
    )


def test_generate(model_meta_data):
    result = metadata.Model.generate(
        "domain",
        "model_id",
        metadata.ModelType.generate("library", "class-name"),
    )
    assert model_meta_data == result


def test_encode_and_decode(model_meta_data):
    # pylint: disable=no-member
    json_result = model_meta_data.to_json()
    result = metadata.Model.from_json(json_result)
    assert result == model_meta_data
