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
import os
from datetime import datetime, timedelta
from pathlib import Path
import pytest

from modelstore.metadata import metadata
from modelstore.storage.local import FileSystemStorage

# pylint: disable=missing-function-docstring


def mock_meta_data(domain: str, model_id: str, inc_time: int) -> metadata.Summary:
    return metadata.Summary.generate(
        model_meta_data=metadata.Model.generate(
            domain=domain,
            model_id=model_id,
            model_type=None,
        ),
        code_meta_data=metadata.Code.generate(
            deps_list=[],
            created=datetime.now() + timedelta(hours=inc_time),
        ),
        storage_meta_data=None,
    )


@pytest.fixture
def mock_model_file(tmp_path):
    model_file = os.path.join(tmp_path, "test-file.txt")
    Path(model_file).touch()
    return model_file


@pytest.fixture
def mock_blob_storage(tmp_path):
    return FileSystemStorage(str(tmp_path))


def assert_file_contents_equals(file_path: str, expected: metadata.Summary):
    # pylint: disable=unspecified-encoding
    # pylint: disable=no-member
    with open(file_path, "r") as lines:
        actual = metadata.Summary.from_json(lines.read())
    assert expected == actual
