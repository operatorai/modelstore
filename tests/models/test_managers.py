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

from modelstore.models import managers
from modelstore.models.catboost import CatBoostManager
from modelstore.models.missingmanager import MissingDepManager
from modelstore.models.pytorch import PyTorchManager
from modelstore.models.sklearn import SKLearnManager
from modelstore.models.xgboost import XGBoostManager


def test_get_manager():
    assert managers.get_manager("sklearn") == SKLearnManager
    assert managers.get_manager("pytorch") == PyTorchManager
    assert managers.get_manager("torch") == PyTorchManager
    assert managers.get_manager("xgboost") == XGBoostManager
    assert managers.get_manager("catboost") == CatBoostManager
    assert managers.get_manager("some-other-dep") == MissingDepManager
