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
from typing import Any

import joblib


def save_json(tmp_dir: str, file_name: str, data: dict) -> str:
    target = os.path.join(tmp_dir, file_name)
    with open(target, "w") as out:
        out.write(json.dumps(data))
    return target


def save_joblib(tmp_dir: str, model: Any, file_name: str) -> str:
    model_path = os.path.join(tmp_dir, file_name)
    joblib.dump(model, model_path)
    return model_path


def load_joblib(model_path: str) -> Any:
    return joblib.load(model_path)
