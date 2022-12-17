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

from modelstore.models import managers
from modelstore.models.catboost import CatBoostManager
from modelstore.models.pytorch import PyTorchManager
from modelstore.models.pytorch_lightning import PyTorchLightningManager
from modelstore.models.sklearn import SKLearnManager
from modelstore.models.tensorflow import TensorflowManager
from modelstore.models.xgboost import XGBoostManager
from modelstore.models.pyspark import PySparkManager

# pylint: disable=missing-function-docstring


def test_iter_libraries():
    mgrs = {library: manager for library, manager in managers.iter_libraries()}
    assert len(mgrs) == 18
    assert isinstance(mgrs["sklearn"], SKLearnManager)
    assert isinstance(mgrs["pytorch"], PyTorchManager)
    assert isinstance(mgrs["xgboost"], XGBoostManager)
    assert isinstance(mgrs["catboost"], CatBoostManager)
    assert isinstance(mgrs["pytorch_lightning"], PyTorchLightningManager)
    assert isinstance(mgrs["pyspark"], PySparkManager)


def test_matching_managers_empty_set():
    with pytest.raises(ValueError):
        managers.matching_managers([], model="none")


def test_no_matching_managers():
    libraries = [m for _, m in managers.iter_libraries()]
    with pytest.raises(ValueError):
        managers.matching_managers(libraries, model="none")


def test_get_keras_manager():
    # The keras manager was merged with the tensorflow one
    # in modelstore==0.0.73; here we test explicitly that
    # modelstore returns the TensorflowManager for
    # backwards compatibility
    manager = managers.get_manager("keras")
    assert isinstance(manager, TensorflowManager)


def test_get_manager():
    # pylint: disable=protected-access
    for name, manager_type in managers._LIBRARIES.items():
        manager = managers.get_manager(name)
        assert isinstance(manager, manager_type)


def test_get_unknown_manager():
    with pytest.raises(KeyError):
        managers.get_manager("an-unknown-library")
