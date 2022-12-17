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

import modelstore
from modelstore.metadata.code.code import Code
from modelstore.metadata.model.model import Model, ModelType, Dataset
from modelstore.metadata.storage.storage import Storage
from modelstore.metadata.utils.utils import exclude_field, validate_json_serializable


@dataclass_json
@dataclass
class Summary:

    """Summary holds all of the fields that are captured
    when a model is saved"""

    model: Model
    storage: Storage
    modelstore: str  # Version of modelstore
    code: Optional[Code] = field(default=None, metadata=config(exclude=exclude_field))
    extra: Optional[dict] = field(default=None, metadata=config(exclude=exclude_field))

    @classmethod
    def generate(
        cls,
        code_meta_data: Code,
        model_meta_data: Model,
        storage_meta_data: Storage,
        extra_metadata: dict = None,
    ) -> "Summary":
        """Generates all of the meta data for a model
        and adds the modelstore version"""
        validate_json_serializable("extra_metadata", extra_metadata)
        return Summary(
            model=model_meta_data,
            storage=storage_meta_data,
            modelstore=modelstore.__version__,
            code=code_meta_data,
            extra=extra_metadata,
        )

    def dumps(self, target_file: str):
        """Dumps the data class as JSON into target_file"""
        # pylint: disable=no-member
        # pylint: disable=unspecified-encoding
        with open(target_file, "w") as out:
            out.write(self.to_json())

    @classmethod
    def loads(cls, source_file: str) -> "Summary":
        """Loads the data class from a JSON source_file"""
        # pylint: disable=no-member
        # pylint: disable=unspecified-encoding
        with open(source_file, "r") as lines:
            content = lines.read()
        return Summary.from_json(content)

    def model_type(self) -> ModelType:
        """Returns the model type"""
        return self.model.model_type

    def dataset(self) -> Dataset:
        """Returns meta data about the training data"""
        return self.model.data
