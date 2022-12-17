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
from typing import Any
import json


def remove_nones(values: dict) -> dict:
    """Removes any entries in a dictionary that have None values"""
    return {k: v for k, v in values.items() if v is not None}


def exclude_field(value: Any) -> bool:
    """Whether to exclude a field from being included in the JSON
    meta data"""
    return value is None


def validate_json_serializable(name: str, value: dict):
    """Validates that `value` is a JSON serializable dictionary"""
    if value is None:
        # None fields will not be dumped from dataclasses
        return
    if not isinstance(value, dict):
        raise TypeError(f"{name} is not a dictionary")
    try:
        # @Future: check if `value` has fields that can be auto-converted
        # to make it JSON serializable (e.g., np.array to list)
        json.dumps(value)
    except Exception as exc:
        raise TypeError(f"{name} must be json serializable") from exc
