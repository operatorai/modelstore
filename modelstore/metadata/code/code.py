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
from typing import Optional
from datetime import datetime
from dataclasses import dataclass, field
from dataclasses_json.cfg import config
from dataclasses_json import dataclass_json

from modelstore.metadata.code import runtime, dependencies, revision
from modelstore.metadata.utils.utils import remove_nones, exclude_field


@dataclass_json
@dataclass
class Code:

    """Code contains fields that are captured about
    the code/runtime when a model is saved"""

    runtime: str
    user: str
    created: str
    dependencies: dict
    git: Optional[dict] = field(default=None, metadata=config(exclude=exclude_field))

    @classmethod
    def generate(cls, deps_list: list, created: datetime = None) -> "Code":
        """Generates the meta data for the code being run to create the model"""
        versioned_deps = dependencies.get_dependency_versions(deps_list)
        if created is None:
            # created can be overridden in unit tests where we need to
            # control time stamps of mock model objects
            created = datetime.now()
        return Code(
            runtime=runtime.get_python_version(),
            user=runtime.get_user(),
            created=created.strftime("%Y/%m/%d/%H:%M:%S"),
            dependencies=remove_nones(versioned_deps),
            git=revision.git_meta(),
        )
