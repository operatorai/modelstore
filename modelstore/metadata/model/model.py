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
from dataclasses import dataclass, field
from dataclasses_json import dataclass_json

from modelstore.metadata.model.model_type import ModelType

@dataclass_json
@dataclass
class Model:

    """ Model contains fields that are captured about
    the model when it is saved """

    domain: str
    model_id: str
    model_type: ModelType
    parameters: dict = field(default_factory=lambda: {})
    data: dict = field(default_factory=lambda: {}) # @TODO this could be a nested dataclass

    @classmethod
    def generate(cls,
        domain: str,
        model_id: str,
        model_type: ModelType,
        parameters: dict = None,
        data: dict = None) -> "Model":
        """ Generates the meta data for the model that is being saved """
        return Model(
            domain=domain,
            model_id=model_id,
            model_type=model_type,
            parameters=parameters,
            data=data
        )
