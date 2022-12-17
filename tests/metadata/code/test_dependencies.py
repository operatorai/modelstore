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
import sys

import pytest
from modelstore.metadata.code import dependencies

# pylint: disable=protected-access
# pylint: disable=missing-function-docstring


def test_get_version():
    assert dependencies._get_version("a-missing-dependency") is None
    assert dependencies._get_version("pytest") == pytest.__version__
    if "isort" in sys.modules:
        # Force import
        del sys.modules["isort"]
    assert dependencies._get_version("isort") == "5.11.3"


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
    result = dependencies.get_dependency_versions(test_deps)
    assert list(result.keys()) == test_deps


def test_module_exists():
    assert dependencies.module_exists("pytest") is True
    assert dependencies.module_exists("a-missing-mod") is False
