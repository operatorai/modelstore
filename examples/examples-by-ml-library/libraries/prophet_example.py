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

import random
from datetime import datetime, timedelta

import pandas as pd
from modelstore.model_store import ModelStore
from prophet import Prophet

_DOMAIN_NAME = "example-prophet-forecast"


def _train_example_model() -> Prophet:
    print("🤖  Creating fake time series data...")
    now = datetime.now()
    rows = []
    for i in range(100):
        rows.append({"ds": now + timedelta(days=i), "y": random.gauss(0, 1)})
    df = pd.DataFrame(rows)

    model = Prophet()
    model.fit(df)

    # Show some predictions
    future = model.make_future_dataframe(periods=5)
    print(f"🔍  Predictions = {future.tail().to_dict(orient='records')}.")
    return model


def train_and_upload(modelstore: ModelStore) -> dict:
    # Train an Annoy index
    model = _train_example_model()

    # Upload the model to the model store
    print(f'⤴️  Uploading the Prophet model to the "{_DOMAIN_NAME}" domain.')
    meta_data = modelstore.upload(
        _DOMAIN_NAME,
        model=model,
    )
    return meta_data


def load_and_test(modelstore: ModelStore, model_domain: str, model_id: str):
    # Load the model back into memory!
    print(f'⤵️  Loading the Prophet "{model_domain}" domain model={model_id}')
    model = modelstore.load(model_domain, model_id)

    # Show some predictions
    future = model.make_future_dataframe(periods=5)
    print(f"🔍  Predictions = {future.tail().to_dict(orient='records')}.")
