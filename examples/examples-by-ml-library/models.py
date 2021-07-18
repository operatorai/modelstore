import catboost as ctb
import keras
from fastai.tabular.all import *
from gensim.models import word2vec
from modelstore.model_store import ModelStore

import lightgbm as lgb
from datasets import (
    load_diabetes_dataframe,
    load_diabetes_dataset,
    load_newsgroup_sentences,
)

_DIABETES_MODEL_DOMAIN = "diabetes-boosting-demo"
_NEWSGROUP_EMBEDDINGS_DOMAIN = "newsgroups-embeddings"


def run_catboost_example(modelstore: ModelStore) -> dict:
    # Load the data
    X_train, _, y_train, _ = load_diabetes_dataset()

    # Train the model
    print("🤖  Training a CatBoostRegressor")
    model = ctb.CatBoostRegressor()
    model.fit(X_train, y_train)

    # Upload the model to the model store
    print(
        f'⤴️  Uploading the catboost model to the "{_DIABETES_MODEL_DOMAIN}" domain.'
    )
    # Alternative: modelstore.catboost.upload(model_domain, model=model)
    return modelstore.upload(_DIABETES_MODEL_DOMAIN, model=model)


def run_fastai_example(modelstore: ModelStore) -> dict:
    # Load the data
    df = load_diabetes_dataframe()

    # Train the model
    print(f"🤖  Training a fastai tabular learner...")
    dl = TabularDataLoaders.from_df(df, y_names=["y"])
    learner = tabular_learner(dl)
    learner.fit_one_cycle(n_epoch=1)

    # Upload the model to the model store
    print(
        f'⤴️  Uploading the fastai model to the "{_DIABETES_MODEL_DOMAIN}" domain.'
    )
    return modelstore.upload(_DIABETES_MODEL_DOMAIN, learner=learner)


def run_gensim_example(modelstore: ModelStore) -> dict:
    # Load the data
    sentences = load_newsgroup_sentences()

    # Train a word2vec model
    print(f"🤖  Training a word2vec model...")
    model = word2vec.Word2Vec(sentences, min_count=2)

    # Upload the model to the model store
    print(
        f"⤴️  Uploading the model to the {_NEWSGROUP_EMBEDDINGS_DOMAIN} domain."
    )
    return modelstore.upload(_NEWSGROUP_EMBEDDINGS_DOMAIN, model=model)


def run_keras_example(modelstore: ModelStore) -> dict:
    # Load the data
    X_train, _, y_train, _ = load_diabetes_dataset()

    # Train a model
    print(f"🤖  Training a keras model...")
    inputs = keras.Input(shape=(10,))
    outputs = keras.layers.Dense(1)(inputs)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="mean_squared_error")
    model.fit(X_train, y_train, epochs=10)

    # Upload the model to the model store
    print(
        f'⤴️  Uploading the keras model to the "{_DIABETES_MODEL_DOMAIN}" domain.'
    )
    return modelstore.upload(_DIABETES_MODEL_DOMAIN, model=model)


def run_lightgbm_example(modelstore: ModelStore) -> dict:
    # Load the data
    X_train, X_test, y_train, y_test = load_diabetes_dataset()

    # Train a model
    print(f"🤖  Training a light GBM model...")
    train_data = lgb.Dataset(X_train, label=y_train)
    validation_data = lgb.Dataset(X_test, y_test)
    num_round = 5
    param = {"num_leaves": 31, "objective": "binary"}
    model = lgb.train(
        param, train_data, num_round, valid_sets=[validation_data]
    )

    # Upload the model to the model store
    print(
        f'⤴️  Uploading the light GBM model to the "{_DIABETES_MODEL_DOMAIN}" domain.'
    )
    return modelstore.upload(_DIABETES_MODEL_DOMAIN, model=model)
