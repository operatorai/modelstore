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
import json
import os

# pylint: disable=redefined-outer-name
TEST_FILE_NAME = "test-file.txt"
TEST_FILE_CONTENTS = json.dumps({"k": "v"})
TEST_FILE_LIST = [f"test-file-{i}.json" for i in range(3)]


def temp_file(tmp_path, contents=TEST_FILE_CONTENTS):
    source = os.path.join(tmp_path, TEST_FILE_NAME)
    with open(source, "w") as out:
        out.write(contents)
    return source


def file_contains_expected_contents(file_path):
    with open(file_path, "r") as lines:
        contents = lines.read()
    return contents == TEST_FILE_CONTENTS


def remote_path():
    return "prefix/to/file/"


def remote_file_path():
    return os.path.join(remote_path(), TEST_FILE_NAME)
