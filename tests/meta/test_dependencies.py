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
import json
import os
import sys

import pytest
from modelstore.meta import dependencies

# pylint: disable=protected-access


def test_get_version():
    assert dependencies._get_version("a-missing-dependency") is None
    assert dependencies._get_version("pytest") == pytest.__version__
    if "isort" in sys.modules:
        # Force import
        del sys.modules["isort"]
    assert dependencies._get_version("isort") == "5.6.4"


def test_get_dependency_versions():
    test_deps = [
        "annoy",
        "pytest",
        "pylint",
        "black",
        "flake8",
        "isort",
        "a-missing-dependency",
        "pickle",
    ]
    expected = {
        "annoy": "1.17.0",
        "black": "20.8b1",
        "pytest": pytest.__version__,
        "pylint": "2.6.0",
        "flake8": "3.8.4",
        "isort": "5.6.4",
        "a-missing-dependency": None,
        "pickle": "4.0",
    }
    result = dependencies.get_dependency_versions(test_deps)
    assert result == expected


def test_module_exists():
    assert dependencies.module_exists("pytest") is True
    assert dependencies.module_exists("a-missing-mod") is False


def test_save_dependencies(tmp_path):
    test_deps = [
        "pytest",
        "pylint",
        "black",
        "flake8",
        "isort",
        "a-missing-dependency",
    ]
    expected = {
        "black": "20.8b1",
        "flake8": "3.8.4",
        "isort": "5.6.4",
        "pytest": pytest.__version__,
        "pylint": "2.6.0",
    }
    tmp_file = dependencies.save_dependencies(tmp_path, test_deps)
    assert os.path.split(tmp_file)[1] == "python-info.json"
    with open(tmp_file, "r") as lines:
        result = json.loads(lines.read())
    assert result == expected
