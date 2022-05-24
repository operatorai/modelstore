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
from datetime import datetime
import pytest

import modelstore

from modelstore.metadata.code.code import CodeMetaData
from modelstore.metadata.model.model import ModelMetaData, ModelTypeMetaData
from modelstore.metadata.storage.storage import StorageMetaData
from modelstore.metadata.metadata import MetaData

# pylint: disable=redefined-outer-name
# pylint: disable=protected-access
# pylint: disable=missing-function-docstring
# pylint: disable=no-member

@pytest.fixture
def meta_data():
    return MetaData(
        code=CodeMetaData(
            runtime="python:1.2.3",
            user="username",
            created=datetime.now().strftime("%Y/%m/%d/%H:%M:%S"),
            dependencies={},
            git={"repository": "test"},
        ),
        model=ModelMetaData.generate(
            domain="domain",
            model_id="model-id",
            model_type=ModelTypeMetaData.generate(
                "library",
                "class-name",
            ),
        ),
        storage=StorageMetaData.from_path(
            "example-storage-type",
            "path-to-files",
        ),
        modelstore=modelstore.__version__,
    )


def test_generate(meta_data):
    result = MetaData.generate(
        code_meta_data=meta_data.code,
        model_meta_data=meta_data.model,
        storage_meta_data=meta_data.storage
    )
    assert result == meta_data

    encoded = result.to_json()
    decoded = MetaData.from_json(encoded)
    assert decoded == meta_data
    assert decoded.code == meta_data.code
    assert decoded.model == meta_data.model
    assert decoded.storage == meta_data.storage


def test_dump_and_load(meta_data, tmp_path):
    target_file = os.path.join(tmp_path, "meta.json")
    assert not os.path.exists(target_file)
    meta_data.dumps(target_file)
    assert os.path.exists(target_file)
    # pylint: disable=bare-except
    # pylint: disable=unspecified-encoding

    loaded = MetaData.loads(target_file)
    assert loaded == meta_data
