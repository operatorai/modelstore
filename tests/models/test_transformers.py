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
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DistilBertForSequenceClassification,
    DistilBertTokenizer,
)
from transformers.file_utils import CONFIG_NAME

from modelstore.models.transformers import (
    TransformersManager,
    _save_transformers,
)

# pylint: disable=protected-access
# pylint: disable=redefined-outer-name


@pytest.fixture
def model_name():
    return "distilbert-base-cased"


@pytest.fixture
def model_config(model_name):
    return AutoConfig.from_pretrained(
        model_name, num_labels=3, finetuning_task="mnli",
    )


@pytest.fixture
def tokenizer(model_name):
    return AutoTokenizer.from_pretrained(model_name)


@pytest.fixture()
def model(model_name, model_config):
    return AutoModelForSequenceClassification.from_pretrained(
        model_name, config=model_config,
    )


@pytest.fixture
def tr_manager():
    return TransformersManager()


def test_name(tr_manager):
    assert tr_manager.name() == "transformers"


def test_model_info(tr_manager):
    exp = {}
    res = tr_manager.model_info()
    assert exp == res


def test_required_kwargs(tr_manager):
    assert tr_manager._required_kwargs() == ["model", "tokenizer"]


def test_get_functions(tr_manager):
    assert (
        len(tr_manager._get_functions(config="c", model="m", tokenizer="t"))
        == 1
    )


def test_save_transformers(model_config, model, tokenizer, tmp_path):
    exp = os.path.join(tmp_path, "transformers")
    file_path = _save_transformers(tmp_path, model_config, model, tokenizer)
    assert exp == file_path

    # Validate config
    config_file = os.path.join(exp, CONFIG_NAME)
    assert os.path.exists(config_file)
    with open(config_file, "r") as lines:
        config_json = json.loads(lines.read())
    assert config_json == json.loads(model_config.to_json_string())

    # Validate model
    model = AutoModelForSequenceClassification.from_pretrained(
        file_path, config=model_config,
    )
    assert isinstance(model, DistilBertForSequenceClassification)

    # Validate tokenizer
    token = AutoTokenizer.from_pretrained(file_path)
    assert isinstance(token, DistilBertTokenizer)
