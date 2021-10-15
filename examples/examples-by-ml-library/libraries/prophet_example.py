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


def load_and_test(modelstore: ModelStore, model_id: str):
    # Load the model back into memory!
    print(f'⤵️  Loading the Prophet "{_DOMAIN_NAME}" domain model={model_id}')
    model = modelstore.load(_DOMAIN_NAME, model_id)

    # Show some predictions
    future = model.make_future_dataframe(periods=5)
    print(f"🔍  Predictions = {future.tail().to_dict(orient='records')}.")
