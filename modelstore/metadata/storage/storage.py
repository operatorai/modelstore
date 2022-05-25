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
from dataclasses import dataclass
from dataclasses_json import dataclass_json

@dataclass_json
@dataclass
class Storage:

    """ Storage contains fields that are captured about
    where the model type is saved """

    type: str # Constant to describe the storage type

    # Path-like storage
    path: str
    
    # Container-like storage
    bucket: str
    container: str # Retained for backwards compatibility (Azure)
    prefix: str

    @classmethod
    def from_path(cls, storage_type: str, path: str) -> "Storage":
        """ Generates the meta data about where the model
        is going to be saved """
        return Storage(
            type=storage_type,
            path=path,
            bucket=None,
            container=None,
            prefix=None,
        )

    @classmethod
    def from_bucket(cls, storage_type: str, bucket: str, prefix: str) -> "Storage":
        """ Generates the meta data about where the model
        is going to be saved """
        return Storage(
            type=storage_type,
            path=None,
            bucket=bucket,
            container=None,
            prefix=prefix,
        )

    @classmethod
    def from_container(cls, storage_type: str, container: str, prefix: str) -> "Storage":
        """ Generates the meta data about where the model
        is going to be saved """
        return Storage(
            type=storage_type,
            path=None,
            bucket=None,
            container=container,
            prefix=prefix,
        )
