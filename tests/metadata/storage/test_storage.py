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
import json 
from modelstore.metadata.storage.storage import StorageMetaData

# pylint: disable=missing-function-docstring
# pylint: disable=redefined-outer-name
# pylint: disable=no-member


def test_generate_from_path():
    expected = StorageMetaData(
        type="file_system",
        path="/path/to/files",
        bucket=None,
        container=None,
        prefix=None,
    )
    result = StorageMetaData.from_path("file_system", "/path/to/files")
    assert expected == result

    result_dict = json.loads(result.to_json())
    assert result_dict["container"] is None
    assert result_dict["type"] == "file_system"

    loaded = StorageMetaData.from_json(result.to_json())
    assert loaded == expected


def test_generate_from_container():
    expected = StorageMetaData(
        type="container-system",
        path=None,
        bucket=None,
        container="container-name",
        prefix="/path/to/files",
    )
    result = StorageMetaData.from_container("container-system", "container-name", "/path/to/files")
    assert expected == result


def test_generate_from_bucket():
    expected = StorageMetaData(
        type="bucket-system",
        path=None,
        bucket="bucket-name",
        container=None,
        prefix="/path/to/files",
    )
    result = StorageMetaData.from_bucket("bucket-system", "bucket-name", "/path/to/files")
    assert expected == result
