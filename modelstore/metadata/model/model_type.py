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
from typing import List, Dict, Optional
import os
from dataclasses import dataclass
from dataclasses_json import dataclass_json

_MODEL_TYPE_FILE = "model-info.json"


@dataclass_json
@dataclass
class ModelType:

    """ ModelTypeMetaData contains fields that are captured about
    the model type when it is saved """

    library: str
    type: str
    models: Optional[List['ModelType']] # Used by the multiple model manager

    @classmethod
    def generate(cls, library: str, class_name: str = None, models: List[Dict] = None) -> "ModelType":
        """ Generates the meta data for the type of model
        that is being saved """
        return ModelType(
            library=library,
            type=class_name,
            models=models,
        )

    def dumps(self, tmp_dir: str) -> str:
        """ Dumps the data class as JSON into a file
        and returns the path to the file """
        # pylint: disable=no-member
        # pylint: disable=unspecified-encoding
        target_file = os.path.join(tmp_dir, _MODEL_TYPE_FILE)
        with open(target_file, "w") as out:
            out.write(self.to_json())
        return target_file
