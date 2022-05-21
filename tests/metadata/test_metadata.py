#    Copyright 2020 Neal Lathia
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
import modelstore
from modelstore.metadata.metadata import MetaData

# pylint: disable=protected-access
# pylint: disable=missing-function-docstring

def test_generate():
    expected = MetaData(
        code=None,
        model=None,
        storage=None,
        modelstore=modelstore.__version__,
    )
    result = MetaData.generate(
        code_meta_data=None,
        model_meta_data=None,
        storage_meta_data=None
    )
    assert result == expected
