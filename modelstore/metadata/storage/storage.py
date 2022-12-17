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
from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
from dataclasses_json.cfg import config

from modelstore.metadata.utils.utils import exclude_field


@dataclass_json
@dataclass
class Storage:

    """Storage contains fields that are captured about
    where the model is saved"""

    # Constant to describe the storage type
    type: str

    # Path-like storage (e.g. local)
    root: Optional[str] = field(default=None, metadata=config(exclude=exclude_field))
    path: Optional[str] = field(default=None, metadata=config(exclude=exclude_field))

    # Container-like storage
    bucket: Optional[str] = field(default=None, metadata=config(exclude=exclude_field))
    prefix: Optional[str] = field(default=None, metadata=config(exclude=exclude_field))

    # Retained for backwards compatibility (Azure)
    container: Optional[str] = field(
        default=None, metadata=config(exclude=exclude_field)
    )

    @classmethod
    def from_path(cls, storage_type: str, root: str, path: str) -> "Storage":
        """Generates the meta data about where the model
        is going to be saved when it is saved in path-like storage"""
        return Storage(
            type=storage_type,
            root=root,
            path=path,
        )

    @classmethod
    def from_bucket(cls, storage_type: str, bucket: str, prefix: str) -> "Storage":
        """Generates the meta data about where the model
        is going to be saved when it is saved in container storage"""
        return Storage(
            type=storage_type,
            bucket=bucket,
            prefix=prefix,
        )

    @classmethod
    def from_container(
        cls, storage_type: str, container: str, prefix: str
    ) -> "Storage":
        """Generates the meta data about where the model
        is going to be saved when it is saved in an Azure container"""
        return Storage(
            type=storage_type,
            container=container,
            prefix=prefix,
        )
