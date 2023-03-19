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
from typing import Any, Optional
from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
from dataclasses_json.cfg import config

from modelstore.metadata.dataset.features import Features
from modelstore.metadata.dataset.labels import Labels
from modelstore.metadata.utils.utils import exclude_field


@dataclass_json
@dataclass
class Dataset:

    """Dataset contains fields that are captured about
    the training dataset when the model is saved"""

    features: Optional[Features] = field(default=None, metadata=config(exclude=exclude_field))
    labels: Optional[Labels] = field(default=None, metadata=config(exclude=exclude_field))

    @classmethod
    def generate(cls, features: Any = None, labels: Any = None) -> "Dataset":
        """Returns summary stats about a dataset"""
        features = Features.generate(features)
        labels = Labels.generate(labels)
        if features is None and labels is None:
            return None
        return Dataset(features=features, labels=labels)
