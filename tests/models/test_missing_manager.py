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
from modelstore.models.missing_manager import MissingDepManager

# pylint: disable=redefined-outer-name


@pytest.fixture
def missing_library_manager():
    return MissingDepManager("some-missing-library")


def test_missing_dep_create(missing_library_manager):
    with pytest.raises(ModuleNotFoundError):
        missing_library_manager.upload("test-domain")


def test_missing_manager_matches_with(missing_library_manager):
    assert not missing_library_manager.matches_with(model="value")


def test_load_model(missing_library_manager):
    with pytest.raises(ModuleNotFoundError):
        missing_library_manager.load("model-path", {})
