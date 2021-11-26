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
import json
import os

import pytest
from modelstore.models.transformers import (
    MODEL_DIRECTORY,
    TransformersManager,
    _save_transformers,
)
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DistilBertForSequenceClassification,
    DistilBertModel,
    DistilBertTokenizerFast,
    PreTrainedTokenizerFast,
)
from transformers.file_utils import CONFIG_NAME

# pylint: disable=protected-access
# pylint: disable=redefined-outer-name


@pytest.fixture
def model_name():
    return "distilbert-base-cased"


@pytest.fixture
def tr_config(model_name):
    return AutoConfig.from_pretrained(
        model_name,
        num_labels=3,
        finetuning_task="mnli",
    )


@pytest.fixture
def tr_tokenizer(model_name):
    return AutoTokenizer.from_pretrained(model_name)


@pytest.fixture()
def tr_model(model_name, tr_config):
    return AutoModelForSequenceClassification.from_pretrained(
        model_name,
        config=tr_config,
    )


@pytest.fixture
def tr_manager():
    return TransformersManager()


def test_model_info(tr_manager):
    exp = {"library": "transformers"}
    res = tr_manager._model_info()
    assert exp == res


@pytest.mark.parametrize(
    "ml_library,should_match",
    [
        ("transformers", True),
        ("xgboost", False),
    ],
)
def test_is_same_library(tr_manager, ml_library, should_match):
    assert tr_manager._is_same_library({"library": ml_library}) == should_match


def test_model_data(tr_manager, tr_model):
    exp = {}
    res = tr_manager._model_data(model=tr_model)
    assert exp == res


def test_required_kwargs(tr_manager):
    assert tr_manager._required_kwargs() == ["model", "tokenizer", "config"]


def test_matches_with(tr_manager, tr_config, tr_model, tr_tokenizer):
    assert tr_manager.matches_with(
        config=tr_config, model=tr_model, tokenizer=tr_tokenizer
    )
    assert not tr_manager.matches_with(
        config=tr_config, model="a-string-value", tokenizer=tr_tokenizer
    )
    assert not tr_manager.matches_with(classifier=tr_model)


def test_get_functions(tr_manager, tr_config, tr_model, tr_tokenizer):
    assert (
        len(
            tr_manager._get_functions(
                config=tr_config, model=tr_model, tokenizer=tr_tokenizer
            )
        )
        == 1
    )


def test_get_params(tr_manager, tr_config):
    exp = tr_config.to_dict()
    res = tr_manager._get_params(config=tr_config)
    assert exp == res


def test_save_transformers(tr_config, tr_model, tr_tokenizer, tmp_path):
    exp = os.path.join(tmp_path, "transformers")
    file_path = _save_transformers(tmp_path, tr_config, tr_model, tr_tokenizer)
    assert exp == file_path

    # Validate config
    config_file = os.path.join(exp, CONFIG_NAME)
    assert os.path.exists(config_file)
    with open(config_file, "r") as lines:
        config_json = json.loads(lines.read())
    assert config_json == json.loads(tr_config.to_json_string())

    # Validate model
    model = AutoModelForSequenceClassification.from_pretrained(
        exp,
        config=tr_config,
    )
    assert isinstance(model, DistilBertForSequenceClassification)

    # Validate tokenizer
    token = AutoTokenizer.from_pretrained(exp)
    assert isinstance(token, PreTrainedTokenizerFast)


def test_load_model(tmp_path, tr_manager, tr_model, tr_config, tr_tokenizer):
    # Save the model to a tmp directory
    model_dir = os.path.join(tmp_path, MODEL_DIRECTORY)
    tr_model.save_pretrained(model_dir)
    tr_config.save_pretrained(model_dir)
    tr_tokenizer.save_pretrained(model_dir)

    #  Load the model
    loaded_model, loaded_tokenizer, loaded_config = tr_manager.load(
        tmp_path, {}
    )

    # Expect the two to be the same
    assert isinstance(loaded_model, DistilBertModel)
    assert isinstance(loaded_config, type(tr_config))
    assert isinstance(loaded_tokenizer, DistilBertTokenizerFast)
