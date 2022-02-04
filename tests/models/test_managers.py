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
from modelstore.models.pytorch import PyTorchManager
from modelstore.models.pytorch_lightning import PyTorchLightningManager
from modelstore.models.sklearn import SKLearnManager
from modelstore.models.xgboost import XGBoostManager


def test_iter_libraries():
    mgrs = {library: manager for library, manager in managers.iter_libraries()}
    assert len(mgrs) == 17
    assert isinstance(mgrs["sklearn"], SKLearnManager)
    assert isinstance(mgrs["pytorch"], PyTorchManager)
    assert isinstance(mgrs["xgboost"], XGBoostManager)
    assert isinstance(mgrs["catboost"], CatBoostManager)
    assert isinstance(mgrs["pytorch_lightning"], PyTorchLightningManager)
