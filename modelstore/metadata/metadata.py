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

import modelstore
from modelstore.metadata.code.code import CodeMetaData
from modelstore.metadata.model.model import ModelMetaData
from modelstore.metadata.storage.storage import StorageMetaData

@dataclass
class MetaData:

    """ MetaData holds all of the fields that are captured
    when a model is saved """

    code: CodeMetaData
    model: ModelMetaData
    storage: StorageMetaData
    modelstore: str # Version of modelstore

    @classmethod
    def generate(cls,
        code_meta_data: CodeMetaData,
        model_meta_data: ModelMetaData,
        storage_meta_data: StorageMetaData
    ) -> "MetaData":
        """ Generates all of the meta data for a model 
        and adds the modelstore version """
        return MetaData(
            code=code_meta_data,
            model=model_meta_data,
            storage=storage_meta_data,
            modelstore=modelstore.__version__,
        )
