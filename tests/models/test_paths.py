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

import joblib
from modelstore.models import common


def test_save_json(tmp_path):
    exp = {"key": "value"}
    target = common.save_json(tmp_path, "data.json", exp)
    with open(target, "r") as lines:
        res = json.loads(lines.read())
    assert exp == res


def test_save_joblib(tmp_path):
    exp = {"key": "value"}
    exp_path = os.path.join(tmp_path, "model.joblib")
    target = common.save_joblib(tmp_path, exp, fn="model.joblib")
    assert os.path.exists(exp_path)
    with open(target, "rb") as f:
        res = joblib.load(f)
    assert exp == res
