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
def model_type_meta_data():
    return metadata.ModelType(
        library="a-library",
        type="a-class-name",
        models=None,
    )


@pytest.fixture
def nested_model_type_meta_data():
    return metadata.ModelType(
        library="multiple-models",
        type=None,
        models=[
            metadata.ModelType.generate("sklearn", "RandomForestClassifier"),
            metadata.ModelType.generate("shap", "TreeExplainer"),
        ],
    )


def test_generate(model_type_meta_data):
    result = metadata.ModelType.generate(
        library="a-library",
        class_name="a-class-name",
    )
    assert model_type_meta_data == result


def test_encode_and_decode(model_type_meta_data):
    # pylint: disable=no-member
    json_result = model_type_meta_data.to_json()
    result = metadata.ModelType.from_json(json_result)
    assert result == model_type_meta_data


def test_generate_multiple_models(nested_model_type_meta_data):
    result = metadata.ModelType.generate(
        "multiple-models",
        models=[
            metadata.ModelType.generate("sklearn", "RandomForestClassifier"),
            metadata.ModelType.generate("shap", "TreeExplainer"),
        ],
    )
    assert nested_model_type_meta_data == result


def test_encode_and_decode_nested(nested_model_type_meta_data):
    # pylint: disable=no-member
    json_result = nested_model_type_meta_data.to_json()
    result = metadata.ModelType.from_json(json_result)
    assert result == nested_model_type_meta_data
