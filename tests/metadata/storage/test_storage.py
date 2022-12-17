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

from modelstore.metadata import metadata

# pylint: disable=missing-function-docstring
# pylint: disable=redefined-outer-name
# pylint: disable=no-member


def test_generate_from_path():
    expected = metadata.Storage(
        type="file_system",
        root="root",
        path="/path/to/files",
        bucket=None,
        container=None,
        prefix=None,
    )
    result = metadata.Storage.from_path(
        "file_system",
        "root",
        "/path/to/files",
    )
    assert expected == result

    result_dict = json.loads(result.to_json())
    assert "container" not in result_dict
    assert result_dict["type"] == "file_system"

    loaded = metadata.Storage.from_json(result.to_json())
    assert loaded == expected


def test_generate_from_container():
    expected = metadata.Storage(
        type="container-system",
        path=None,
        bucket=None,
        container="container-name",
        prefix="/path/to/files",
    )
    result = metadata.Storage.from_container(
        "container-system", "container-name", "/path/to/files"
    )
    assert expected == result


def test_generate_from_bucket():
    expected = metadata.Storage(
        type="bucket-system",
        path=None,
        bucket="bucket-name",
        container=None,
        prefix="/path/to/files",
    )
    result = metadata.Storage.from_bucket(
        "bucket-system", "bucket-name", "/path/to/files"
    )
    assert expected == result
