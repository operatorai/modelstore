import json

from modelstore import ModelStore

from datasets import load_diabetes_dataframe
from fastai.tabular.all import *


def train():
    df = load_diabetes_dataframe()

    print(f"🤖  Training a fastai tabular learner...")
    dl = TabularDataLoaders.from_df(df, y_names=["y"])
    learner = tabular_learner(dl)

    learner.fit_one_cycle(n_epoch=1)
    # Skipped for brevity (but important!) evaluate the model
    return learner


def run_fastai_example(modelstore: ModelStore) -> dict:
    model_domain = "diabetes-boosting-demo"

    # In this demo, we train a single layer net
    # Replace this with the code to train your own model
    learner = train()

    # Upload the archive to the model store
    # The first string is the model's domain - which helps you to group
    # many models that are trained on the same target together
    print(f"⤴️  Uploading the archive to the {model_domain} domain.")
    return modelstore.upload(model_domain, learner=learner)
