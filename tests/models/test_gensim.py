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

import pytest
from gensim.models import word2vec
from gensim.test.utils import common_texts
from modelstore.models.gensim import GensimManager

# pylint: disable=protected-access
# pylint: disable=redefined-outer-name


@pytest.fixture
def word2vec_model():
    return word2vec.Word2Vec(common_texts, min_count=1)


@pytest.fixture
def gensim_manager():
    return GensimManager()


@pytest.mark.parametrize(
    "model_type,expected",
    [
        (
            word2vec.Word2Vec,
            {"library": "gensim", "type": "Word2Vec"},
        ),
    ],
)
def test_model_info(gensim_manager, model_type, expected):
    res = gensim_manager._model_info(model=model_type())
    assert expected == res


@pytest.mark.parametrize(
    "ml_library,should_match",
    [
        ("gensim", True),
        ("sklearn", False),
    ],
)
def test_is_model_type(gensim_manager, ml_library, should_match):
    assert (
        gensim_manager._is_model_type({"library": ml_library}) == should_match
    )


def test_model_data(gensim_manager, word2vec_model):
    res = gensim_manager._model_data(model=word2vec_model)
    assert {} == res


def test_required_kwargs(gensim_manager):
    assert gensim_manager._required_kwargs() == ["model"]


def test_matches_with(gensim_manager, word2vec_model):
    assert gensim_manager.matches_with(model=word2vec_model)
    assert not gensim_manager.matches_with(model="a-string-value")
    assert not gensim_manager.matches_with(gensim_model=word2vec_model)


def test_get_functions(gensim_manager, word2vec_model):
    assert len(gensim_manager._get_functions(model=word2vec_model)) == 2
    with pytest.raises(TypeError):
        gensim_manager._get_functions(model="not-a-gensim-model")


@pytest.mark.parametrize(
    "model_type",
    [
        word2vec.Word2Vec,
    ],
)
def test_get_params(gensim_manager, model_type):
    try:
        result = gensim_manager._get_params(model=model_type())
        json.dumps(result)
    except Exception as e:
        pytest.fail(f"Exception when dumping params: {str(e)}")


def test_load_model(gensim_manager):
    # Placeholder - to be implemented
    with pytest.raises(NotImplementedError):
        gensim_manager.load("model-path", {})
