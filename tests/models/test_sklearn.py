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
import pytest
from sklearn.ensemble import GradientBoostingRegressor

from modelstore.models.sklearn import SKLearnManager

# pylint: disable=protected-access


@pytest.fixture
def sklearn_model():
    params = {
        "n_estimators": 500,
        "max_depth": 4,
        "min_samples_split": 5,
        "learning_rate": 0.01,
        "loss": "ls",
    }
    return GradientBoostingRegressor(**params)


def test_required_kwargs():
    mngr = SKLearnManager()
    assert mngr._required_kwargs() == ["model"]


def test_get_functions(sklearn_model):
    mngr = SKLearnManager()
    assert len(mngr._get_functions(model=sklearn_model)) == 1
    with pytest.raises(TypeError):
        mngr._get_functions(model="not-an-sklearn-model")
