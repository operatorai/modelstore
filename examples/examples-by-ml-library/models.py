import catboost as ctb
from modelstore.model_store import ModelStore

from datasets import load_diabetes_dataframe, load_diabetes_dataset
from fastai.tabular.all import *


def run_catboost_example(modelstore: ModelStore) -> dict:
    """
    This function shows an example of training a CatBoostRegressor
    and uploading it to the model store
    """

    # Train the model
    print("🤖  Training a CatBoostRegressor")
    X_train, y_train = load_diabetes_dataset()
    model = ctb.CatBoostRegressor()
    model.fit(X_train, y_train)

    # Upload the model to the model store
    model_domain = "diabetes-boosting-demo"
    print(f'⤴️  Uploading the catboost model to the "{model_domain}" domain.')
    # Alternative: modelstore.catboost.upload(model_domain, model=model)
    return modelstore.upload(model_domain, model=model)


def run_fastai_example(modelstore: ModelStore) -> dict:
    """
    This function shows an example of training a CatBoostRegressor
    and uploading it to the model store
    """

    # Train the model
    print(f"🤖  Training a fastai tabular learner...")
    df = load_diabetes_dataframe()

    dl = TabularDataLoaders.from_df(df, y_names=["y"])
    learner = tabular_learner(dl)
    learner.fit_one_cycle(n_epoch=1)

    # Upload the model to the model store
    model_domain = "diabetes-boosting-demo"
    print(f'⤴️  Uploading the fastai model to the "{model_domain}" domain.')
    # Alternative: modelstore.fastai.upload(model_domain, learner=learner)
    return modelstore.upload(model_domain, learner=learner)
