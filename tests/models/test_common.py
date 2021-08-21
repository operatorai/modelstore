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

import pytest
from modelstore.models import common

# pylint: disable=redefined-outer-name


@pytest.fixture
def value_to_save():
    return {"key": "value"}


def test_save_json(tmp_path, value_to_save):
    target = common.save_json(tmp_path, "data.json", value_to_save)
    with open(target, "r") as lines:
        res = json.loads(lines.read())
    assert value_to_save == res


def test_save_joblib(tmp_path, value_to_save):
    exp_path = os.path.join(tmp_path, "model.joblib")
    # Save returns the full path
    target = common.save_joblib(
        tmp_path, value_to_save, file_name="model.joblib"
    )
    assert target == exp_path
    assert os.path.exists(exp_path)

    # Load takes the full path
    res = common.load_joblib(exp_path)
    assert value_to_save == res
