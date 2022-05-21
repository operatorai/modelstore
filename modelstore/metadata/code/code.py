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
from dataclasses import dataclass

from modelstore.metadata.code import runtime, dependencies, revision
from modelstore.metadata.utils.dicts import remove_nones


@dataclass
class CodeMetaData:

    """ CodeMetaData contains fields that are captured about
    the code/runtime when it is saved """

    runtime: str
    user: str
    created: str
    dependencies: dict
    git: dict


def generate(deps_list: list) -> CodeMetaData:
    """ Generates the meta data for the code being run to create the model """
    versioned_deps = dependencies.get_dependency_versions(deps_list)
    return CodeMetaData(
        runtime=f"python:{runtime.get_python_version()}",
        user=runtime.get_user(),
        created=datetime.now().strftime("%Y/%m/%d/%H:%M:%S"),
        dependencies=remove_nones(versioned_deps),
        git=revision.git_meta()
    )
