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
from datetime import datetime

from mock import patch
import pytest

from modelstore.metadata import metadata

# pylint: disable=missing-function-docstring
# pylint: disable=redefined-outer-name


@pytest.fixture
def now():
    return datetime.now()


@pytest.fixture
def code_meta_data(now):
    return metadata.Code(
        runtime="python:1.2.3",
        user="username",
        created=now.strftime("%Y/%m/%d/%H:%M:%S"),
        dependencies={},
        git={"repository": "test"},
    )


@patch("modelstore.metadata.code.code.revision")
@patch("modelstore.metadata.code.code.runtime")
def test_generate(mock_runtime, mock_revision, code_meta_data, now):
    mock_runtime.get_user.return_value = "username"
    mock_runtime.get_python_version.return_value = "python:1.2.3"
    mock_revision.git_meta.return_value = {"repository": "test"}
    result = metadata.Code.generate([], created=now)
    assert code_meta_data == result


def test_encode_and_decode(code_meta_data):
    # pylint: disable=no-member
    json_result = code_meta_data.to_json()
    result = metadata.Code.from_json(json_result)
    assert result == code_meta_data
