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

from modelstore.storage.util import paths

# pylint: disable=protected-access


def test_create_path():
    prefix = datetime.now().strftime("%Y/%m/%d/%H:%M:%S")
    exp = os.path.join(paths._ROOT, "domain", prefix, "file-name")
    res = paths.get_archive_path("domain", "path/to/file-name")
    assert exp == res


def test_create_metadata_path():
    exp = os.path.join(paths._ROOT, "domain", "versions", "model-id.json")
    res = paths.get_metadata_path("domain", "model-id")
    assert exp == res


def test_create_latest_path():
    exp = os.path.join(paths._ROOT, "domains", "domain.json")
    res = paths.get_domain_path("domain")
    assert exp == res
