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
from modelstore.metadata.model import model

# pylint: disable=missing-function-docstring

def test_generate():
    expected = model.ModelMetaData(
        domain="domain",
        model_id="model_id",
        model_type=model.ModelTypeMetaData("library", "class-name"),
        parameters=None,
        data=None
    )
    result = model.generate(
        "domain",
        "model_id",
        model.ModelTypeMetaData("library", "class-name"),
    )
    assert expected == result
