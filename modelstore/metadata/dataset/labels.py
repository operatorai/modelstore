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
from typing import List, Any, Optional
from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
from dataclasses_json.cfg import config
import numpy as np

from modelstore.metadata.dataset.types import (
    is_numpy_array,
    is_pandas_dataframe,
    is_pandas_series,
)
from modelstore.metadata.utils.utils import exclude_field
from modelstore.utils.log import logger


@dataclass_json
@dataclass
class Labels:

    """Labels contains fields that are captured about
    the training dataset's labels when the model is saved"""

    shape: Optional[List[int]] = field(default=None, metadata=config(exclude=exclude_field))
    values: Optional[dict] = field(default=None, metadata=config(exclude=exclude_field))

    @classmethod
    def generate(cls, values: Any = None) -> "Labels":
        """Returns summary stats about a set of labels"""
        if values is None:
            return None
        if is_numpy_array(values):
            if values.ndim == 1:
                # Array has one dimension (e.g., labels); return its
                # its shape and value counts
                unique, counts = np.unique(values, return_counts=True)
                return Labels(
                    shape=list(values.shape), values=dict(zip(unique, counts))
                )
            # Array is multi-dimensional, only return its shape
            return Labels(
                shape=list(values.shape),
                values=None,
            )
        if is_pandas_dataframe(values):
            # Data frame can have multiple dimensions; only
            # return its shape
            return Labels(
                shape=list(values.shape),
                values=None,
            )
        if is_pandas_series(values):
            # Data series has one dimension (e.g., labels); return
            # its shape and value counts
            return Labels(
                shape=list(values.shape),
                values=values.value_counts().to_dict(),
            )
        logger.debug("Trying to describe unknown type: %s", type(values))
        return None
