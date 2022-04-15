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
from pathlib import PosixPath

import pytest
from modelstore.model_store import ModelStore
from modelstore.utils.exceptions import (
    ModelNotFoundException,
    ModelNotFoundException,
)


def test_model_not_found(tmp_path: PosixPath):
    store = ModelStore.from_file_system(root_directory=str(tmp_path))
    with pytest.raises(ModelNotFoundException):
        store.get_model_info("missing-domain", "missing-model")
