import catboost as ctb
from fastai.tabular.all import *
from gensim.models import word2vec
from modelstore.model_store import ModelStore

import keras
from datasets import (
    load_diabetes_dataframe,
    load_diabetes_dataset,
    load_newsgroup_sentences,
)


def run_catboost_example(modelstore: ModelStore) -> dict:
    # Load the data
    X_train, y_train = load_diabetes_dataset()

    # Train the model
    print("🤖  Training a CatBoostRegressor")
    model = ctb.CatBoostRegressor()
    model.fit(X_train, y_train)

    # Upload the model to the model store
    model_domain = "diabetes-boosting-demo"
    print(f'⤴️  Uploading the catboost model to the "{model_domain}" domain.')
    # Alternative: modelstore.catboost.upload(model_domain, model=model)
    return modelstore.upload(model_domain, model=model)


def run_fastai_example(modelstore: ModelStore) -> dict:
    # Load the data
    df = load_diabetes_dataframe()

    # Train the model
    print(f"🤖  Training a fastai tabular learner...")
    dl = TabularDataLoaders.from_df(df, y_names=["y"])
    learner = tabular_learner(dl)
    learner.fit_one_cycle(n_epoch=1)

    # Upload the model to the model store
    model_domain = "diabetes-boosting-demo"
    print(f'⤴️  Uploading the fastai model to the "{model_domain}" domain.')
    return modelstore.upload(model_domain, learner=learner)


def run_gensim_example(modelstore: ModelStore) -> dict:
    # Load the data
    sentences = load_newsgroup_sentences()

    # Train a word2vec model
    print(f"🤖  Training a word2vec model...")
    model = word2vec.Word2Vec(sentences, min_count=2)

    # Upload the model to the model store
    model_domain = "newsgroups-embeddings"
    print(f"⤴️  Uploading the model to the {model_domain} domain.")
    return modelstore.upload(model_domain, model=model)


def run_keras_example(modelstore: ModelStore) -> dict:
    # Load the data
    X_train, y_train = load_diabetes_dataset()

    # Train a model
    inputs = keras.Input(shape=(10,))
    outputs = keras.layers.Dense(1)(inputs)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="mean_squared_error")

    model.fit(X_train, y_train, epochs=10)

    # Upload the model to the model store
    model_domain = "diabetes-boosting-demo"
    print(f'⤴️  Uploading the keras model to the "{model_domain}" domain.')
    return modelstore.upload(model_domain, model=model)
