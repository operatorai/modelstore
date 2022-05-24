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
from dataclasses import dataclass
from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class ModelTypeMetaData:

    """ ModelTypeMetaData contains fields that are captured about
    the model type when it is saved """

    library: str
    type: str
    models: Optional[List['ModelTypeMetaData']] # Used by the multiple model manager

    @classmethod
    def generate(cls, library: str, class_name: str = None, models: List[Dict] = None) -> "ModelTypeMetaData":
        """ Generates the meta data for the type of model
        that is being saved """
        return ModelTypeMetaData(
            library=library,
            type=class_name,
            models=models,
        )
