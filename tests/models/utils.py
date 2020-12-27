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
import pytest
from sklearn.datasets import make_classification


@pytest.fixture
def classification_data():
    X_train, y_train = make_classification(
        n_features=5, n_redundant=0, n_informative=3, n_clusters_per_class=1
    )
    return X_train, y_train
