#    Copyright 2022 Neal Lathia
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
from typing import List
import os
import json


def metadata() -> dict:
    """Returns a dictionary that contains extra metadata
    to upload alongside the model"""
    return {
        "field_name": "value",
    }


def files(tmp_dir, num_files: int = 2) -> List[str]:
    """Returns the paths to files that contain
    extra data to upload alongside the model"""
    results = []
    for i in range(num_files):
        result = os.path.join(tmp_dir, f"result-{i}.json")
        # pylint: disable=unspecified-encoding
        with open(result, "w") as out:
            out.write(
                json.dumps(
                    {
                        "field-1": "value-1",
                        "field-2": "value-2",
                    }
                )
            )
        results.append(result)
    return results
