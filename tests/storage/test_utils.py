#    Copyright 2021 Neal Lathia
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
from tempfile import TemporaryDirectory
import json
import os

# pylint: disable=redefined-outer-name
# pylint: disable=missing-function-docstring
TEST_FILE_NAME = "test-file.txt"
TEST_FILE_CONTENTS = json.dumps({"k": "v"})
TEST_FILE_LIST = [f"test-file-{i}.json" for i in range(3)]
TEST_FILE_TYPES = ["json", "txt"]


def create_file(tmp_path, contents: str = None) -> str:
    # pylint: disable=unspecified-encoding
    if contents is None:
        contents = TEST_FILE_CONTENTS
    source = os.path.join(tmp_path, TEST_FILE_NAME)
    with open(source, "w") as out:
        out.write(contents)
    return source


def file_contains_expected_contents(file_path: str) -> bool:
    # pylint: disable=unspecified-encoding
    with open(file_path, "r") as lines:
        contents = lines.read()
    return contents == TEST_FILE_CONTENTS


def remote_path() -> str:
    return "prefix/to/file"


def remote_file_path() -> str:
    return os.path.join(remote_path(), TEST_FILE_NAME)


def push_temp_file(storage, contents: str = None) -> str:
    with TemporaryDirectory() as tmp_dir:
        # pylint: disable=protected-access
        result = storage._push(
            create_file(tmp_dir, contents),
            remote_file_path(),
        )
    return result


def push_temp_files(storage, prefix, file_types=TEST_FILE_TYPES):
    with TemporaryDirectory() as tmp_dir:
        for file_type in file_types:
            file_name = f"test-file-source.{file_type}"
            file_path = os.path.join(tmp_dir, file_name)
            # pylint: disable=unspecified-encoding
            with open(file_path, "w") as out:
                out.write(json.dumps({"key": "value"}))

            # Push the file to storage
            # pylint: disable=protected-access
            result = storage._push(file_path, os.path.join(prefix, file_name))
            assert result == os.path.join(prefix, file_name)
