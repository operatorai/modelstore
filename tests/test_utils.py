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
from functools import partial
from pathlib import PosixPath, Path
import os

import pytest
from modelstore.model_store import ModelStore
from modelstore.models.managers import _LIBRARIES
from modelstore.models.missing_manager import MissingDepManager
from modelstore.models.model_manager import ModelManager

# pylint: disable=protected-access
# pylint: disable=redefined-outer-name
# pylint: disable=missing-function-docstring


def libraries_without_sklearn():
    libraries = _LIBRARIES.copy()
    libraries.pop("sklearn")
    return libraries


def iter_only_sklearn(_):
    for k, v in _LIBRARIES.items():
        if k == "sklearn":
            yield k, v()
        else:
            yield k, partial(MissingDepManager, library=k)()


def validate_library_attributes(store: ModelStore, allowed: list, not_allowed: list):
    # During dev mode, all libraries will be installed
    for library in allowed:
        assert hasattr(store, library)
        mgr = store.__getattribute__(library)
        assert issubclass(type(mgr), ModelManager)
        assert not isinstance(mgr, MissingDepManager)

    for library in not_allowed:
        assert hasattr(store, library)
        mgr = store.__getattribute__(library)
        assert issubclass(type(mgr), ModelManager)
        assert isinstance(mgr, MissingDepManager)
        with pytest.raises(ModuleNotFoundError):
            mgr.upload(domain="test", model_id="model-id", model="test")


@pytest.fixture
def model_file(tmp_path: PosixPath):
    file_path = os.path.join(tmp_path, "model.txt")
    Path(file_path).touch()
    return file_path
