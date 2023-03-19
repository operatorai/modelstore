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
from typing import Any, List, Optional

from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
from dataclasses_json.cfg import config

from modelstore.metadata.dataset.types import is_numpy_array, is_pandas_dataframe
from modelstore.metadata.utils.utils import exclude_field
from modelstore.utils.log import logger


@dataclass_json
@dataclass
class Features:

    """Features contains fields that are captured about
    the training dataset's features when the model is saved"""

    shape: Optional[List[int]] = field(default=None, metadata=config(exclude=exclude_field))

    @classmethod
    def generate(cls, values: Any = None) -> "Features":
        """Returns summary stats about a set of features"""
        if values is None:
            return None
        if is_numpy_array(values):
            return Features(
                shape=list(values.shape),
            )
        if is_pandas_dataframe(values):
            return Features(
                shape=list(values.shape),
            )
        logger.debug("Trying to describe unknown type: %s", type(values))
        return None
