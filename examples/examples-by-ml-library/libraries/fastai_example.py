from fastai.tabular.all import *
from modelstore.model_store import ModelStore

from libraries.util.datasets import load_diabetes_dataframe
from libraries.util.domains import DIABETES_DOMAIN


def _train_example_model() -> TabularLearner:
    # Load the data
    df = load_diabetes_dataframe()

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


def load_and_test(modelstore: ModelStore, model_id: str):
    # Load the model back into memory!
    print(f'⤵️  Loading the fastai "{DIABETES_DOMAIN}" domain model={model_id}')
    model = modelstore.load(DIABETES_DOMAIN, model_id)
    # ... use for inference
