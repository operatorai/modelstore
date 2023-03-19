#    Copyright 2023 Neal Lathia
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

from fastai.tabular.all import *
from modelstore.model_store import ModelStore

from libraries.util.datasets import load_regression_dataframe
from libraries.util.domains import DIABETES_DOMAIN


def _train_example_model() -> TabularLearner:
    # Load the data
    df = load_regression_dataframe()

    # Train the model
    print(f"🤖  Training a fastai tabular learner...")
    dl = TabularDataLoaders.from_df(df, y_names=["y"])
    learner = tabular_learner(dl)
    learner.fit_one_cycle(n_epoch=1)
    return learner


def train_and_upload(modelstore: ModelStore) -> dict:
    # Train a fast.ai model
    learner = _train_example_model()

    # Upload the model to the model store
    print(f'⤴️  Uploading the fastai model to the "{DIABETES_DOMAIN}" domain.')
    meta_data = modelstore.upload(DIABETES_DOMAIN, learner=learner)
    return meta_data


def load_and_test(modelstore: ModelStore, model_domain: str, model_id: str):
    # Load the model back into memory!
    print(f'⤵️  Loading the fastai "{model_domain}" domain model={model_id}')
    model = modelstore.load(model_domain, model_id)
    # ... use for inference
