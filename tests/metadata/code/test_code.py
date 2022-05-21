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
from unittest.mock import patch
from datetime import datetime

from modelstore.metadata.code.code import CodeMetaData

# pylint: disable=missing-function-docstring


@patch("modelstore.metadata.code.code.revision")
@patch("modelstore.metadata.code.code.runtime")
def test_generate(mock_runtime, mock_revision):
    mock_runtime.get_user.return_value = "username"
    mock_runtime.get_python_version.return_value = "1.2.3"
    mock_revision.git_meta.return_value = {"repository": "test"}
    expected = CodeMetaData(
        runtime="python:1.2.3",
        user="username",
        created=datetime.now().strftime("%Y/%m/%d/%H:%M:%S"),
        dependencies={},
        git={"repository": "test"},
    )
    result = CodeMetaData.generate([])
    assert expected == result
