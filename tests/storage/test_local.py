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
import os

import pytest
from modelstore.storage.local import FileSystemStorage

# pylint: disable=protected-access
# pylint: disable=redefined-outer-name


@pytest.fixture
def fs_model_store(tmp_path):
    return FileSystemStorage(root_path=str(tmp_path))


def test_validate(fs_model_store):
    assert fs_model_store.validate()
    assert os.path.exists(fs_model_store.root_dir)


def test_list_versions_missing_domain(fs_model_store):
    versions = fs_model_store.list_versions("domain-that-doesnt-exist")
    assert len(versions) == 0
