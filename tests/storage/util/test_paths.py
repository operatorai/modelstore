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
from modelstore.storage.util import paths

# pylint: disable=protected-access


def test_get_archive_path():
    prefix = datetime.now().strftime("%Y/%m/%d/%H:%M:%S")
    exp = os.path.join(paths.MODELSTORE_ROOT, "domain", prefix, "file-name")
    res = paths.get_archive_path("domain", "path/to/file-name")
    assert exp == res


def test_get_versions_path():
    exp = os.path.join(paths.MODELSTORE_ROOT, "example-domain", "versions")
    res = paths.get_versions_path("example-domain")
    assert exp == res


def test_get_versions_path_with_state():
    exp = os.path.join(
        paths.MODELSTORE_ROOT, "example-domain", "versions", "prod"
    )
    res = paths.get_versions_path("example-domain", "prod")
    assert exp == res


def test_get_domains_path():
    exp = os.path.join(paths.MODELSTORE_ROOT, "domains")
    res = paths.get_domains_path()
    assert exp == res


def test_get_domain_path():
    exp = os.path.join(paths.MODELSTORE_ROOT, "domains", "domain.json")
    res = paths.get_domain_path("domain")
    assert exp == res


def test_get_model_states_path():
    exp = os.path.join(paths.MODELSTORE_ROOT, "model_states")
    res = paths.get_model_states_path()
    assert exp == res


def test_get_model_state_path():
    exp = os.path.join(paths.MODELSTORE_ROOT, "model_states", "prod.json")
    res = paths.get_model_state_path("prod")
    assert exp == res


@pytest.mark.parametrize(
    "state_name,is_valid",
    [
        (None, False),
        ("", False),
        ("a", False),
        ("path/to/place", False),
        ("other", True),
    ],
)
def test_is_valid_state_name(state_name, is_valid):
    assert paths.is_valid_state_name(state_name) == is_valid
