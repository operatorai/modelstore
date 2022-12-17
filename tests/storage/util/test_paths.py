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
# pylint: disable=missing-function-docstring


@pytest.mark.parametrize("has_root_prefix", [(True), (False)])
def test_get_archive_path(tmp_path, has_root_prefix):
    root = str(tmp_path) if has_root_prefix else ""
    prefix = datetime.now().strftime("%Y.%m.%d-%H.%M.%S")
    exp = os.path.join(
        root, paths.MODELSTORE_ROOT_PREFIX, "domain", prefix, "model-id", "file-name"
    )
    res = paths.get_archive_path(root, "domain", "model-id", "path/to/file-name")
    assert exp == res


@pytest.mark.parametrize("has_root_prefix", [(True), (False)])
def test_get_model_versions_path_no_root_prefix(tmp_path, has_root_prefix):
    root = str(tmp_path) if has_root_prefix else ""
    exp = os.path.join(
        root,
        paths.MODELSTORE_ROOT_PREFIX,
        "domain",
        "versions",
    )
    res = paths.get_model_versions_path(root, "domain")
    assert exp == res


@pytest.mark.parametrize("has_root_prefix", [(True), (False)])
def test_get_model_versions_path_with_state(tmp_path, has_root_prefix):
    root = str(tmp_path) if has_root_prefix else ""
    exp = os.path.join(
        root,
        paths.MODELSTORE_ROOT_PREFIX,
        "domain",
        "versions",
        "prod",
    )
    res = paths.get_model_versions_path(root, "domain", "prod")
    assert exp == res


@pytest.mark.parametrize("has_root_prefix", [(True), (False)])
def test_get_model_version_path(tmp_path, has_root_prefix):
    root = str(tmp_path) if has_root_prefix else ""
    exp = os.path.join(
        root,
        paths.MODELSTORE_ROOT_PREFIX,
        "domain",
        "versions",
        "model-id.json",
    )
    res = paths.get_model_version_path(root, "domain", "model-id")
    assert exp == res


@pytest.mark.parametrize("has_root_prefix", [(True), (False)])
def test_get_model_version_with_state_path(tmp_path, has_root_prefix):
    root = str(tmp_path) if has_root_prefix else ""
    exp = os.path.join(
        root,
        paths.MODELSTORE_ROOT_PREFIX,
        "domain",
        "versions",
        "prod",
        "model-id.json",
    )
    res = paths.get_model_version_path(
        root,
        "domain",
        "model-id",
        state_name="prod",
    )
    assert exp == res


@pytest.mark.parametrize("has_root_prefix", [(True), (False)])
def test_get_domains_path(tmp_path, has_root_prefix):
    root = str(tmp_path) if has_root_prefix else ""
    exp = os.path.join(
        root,
        paths.MODELSTORE_ROOT_PREFIX,
        "domains",
    )
    res = paths.get_domains_path(root)
    assert exp == res


@pytest.mark.parametrize("has_root_prefix", [(True), (False)])
def test_get_domain_path(tmp_path, has_root_prefix):
    root = str(tmp_path) if has_root_prefix else ""
    exp = os.path.join(
        root,
        paths.MODELSTORE_ROOT_PREFIX,
        "domains",
        "domain.json",
    )
    res = paths.get_domain_path(root, "domain")
    assert exp == res


@pytest.mark.parametrize("has_root_prefix", [(True), (False)])
def test_get_model_states_path(tmp_path, has_root_prefix):
    root = str(tmp_path) if has_root_prefix else ""
    exp = os.path.join(
        root,
        paths.MODELSTORE_ROOT_PREFIX,
        "model_states",
    )
    res = paths.get_model_states_path(root)
    assert exp == res


@pytest.mark.parametrize("has_root_prefix", [(True), (False)])
def test_get_model_state_path(tmp_path, has_root_prefix):
    root = str(tmp_path) if has_root_prefix else ""
    exp = os.path.join(
        root,
        paths.MODELSTORE_ROOT_PREFIX,
        "model_states",
        "prod.json",
    )
    res = paths.get_model_state_path(root, "prod")
    assert exp == res
