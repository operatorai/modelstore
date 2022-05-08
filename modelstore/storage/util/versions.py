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
from datetime import datetime


def sort_by_version(meta_data: dict):
    """Extracts the version from a model's meta data"""
    if "code" in meta_data:
        return datetime.strptime(meta_data["code"]["created"], "%Y/%m/%d/%H:%M:%S")
    if "meta" in meta_data:
        return datetime.strptime(meta_data["meta"]["created"], "%Y/%m/%d/%H:%M:%S")
    return 1


def sorted_by_created(versions: list):
    """Sorts a list of models by version"""
    return sorted(
        versions,
        key=sort_by_version,
        reverse=True,
    )
