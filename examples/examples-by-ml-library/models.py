from tempfile import mkdtemp

import catboost as ctb
import keras
import lightgbm as lgb
import pytorch_lightning as pl
import tensorflow as tf
import xgboost as xgb
from fastai.tabular.all import *
from gensim.models import word2vec
from modelstore.model_store import ModelStore
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from datasets import (
    load_diabetes_dataframe,
    load_diabetes_dataset,
    load_newsgroup_sentences,
)
from networks import ExampleLightningNet, ExampleNet

# pylint: disable=invalid-name
_DIABETES_DOMAIN = "diabetes-boosting-demo"
_NEWSGROUP_EMBEDDINGS_DOMAIN = "newsgroups-embeddings"


def run_catboost_example(modelstore: ModelStore) -> dict:
    # Load the data
    X_train, _, y_train, _ = load_diabetes_dataset()

    # Train the model
    print("🤖  Training a CatBoostRegressor")
    model = ctb.CatBoostRegressor(allow_writing_files=False)
    model.fit(X_train, y_train)

    # Upload the model to the model store
    print(
        f'⤴️  Uploading the catboost model to the "{_DIABETES_DOMAIN}" domain.'
    )
    # Alternative: modelstore.catboost.upload(model_domain, model=model)
    return modelstore.upload(_DIABETES_DOMAIN, model=model)


def run_fastai_example(modelstore: ModelStore) -> dict:
    # Load the data
    df = load_diabetes_dataframe()

    # Train the model
    print(f"🤖  Training a fastai tabular learner...")
    dl = TabularDataLoaders.from_df(df, y_names=["y"])
    learner = tabular_learner(dl)
    learner.fit_one_cycle(n_epoch=1)

    # Upload the model to the model store
    print(f'⤴️  Uploading the fastai model to the "{_DIABETES_DOMAIN}" domain.')
    return modelstore.upload(_DIABETES_DOMAIN, learner=learner)


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
    print(f'⤴️  Uploading the keras model to the "{_DIABETES_DOMAIN}" domain.')
    return modelstore.upload(_DIABETES_DOMAIN, model=model)


def run_lightgbm_example(modelstore: ModelStore) -> dict:
    # Load the data
    X_train, X_test, y_train, y_test = load_diabetes_dataset()

    # Train a model
    print("🤖  Training a light GBM model...")
    train_data = lgb.Dataset(X_train, label=y_train)
    validation_data = lgb.Dataset(X_test, y_test)
    num_round = 5
    param = {"num_leaves": 31, "objective": "binary"}
    model = lgb.train(
        param, train_data, num_round, valid_sets=[validation_data]
    )

    # Upload the model to the model store
    print(
        f'⤴️  Uploading the light GBM model to the "{_DIABETES_DOMAIN}" domain.'
    )
    return modelstore.upload(_DIABETES_DOMAIN, model=model)


def run_pytorch_example(modelstore: ModelStore) -> dict:
    # Load the data
    X_train, _, y_train, _ = load_diabetes_dataset(as_numpy=True)

    # Train the model
    model = ExampleNet()
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())

    for epoch in range(5):
        print(f"🤖  Training epoch: {epoch}...")
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

    # Upload the model to the model store
    print(
        f'⤴️  Uploading the pytorch model to the "{_DIABETES_DOMAIN}" domain.'
    )
    return modelstore.upload(_DIABETES_DOMAIN, model=model, optimizer=optimizer)


def run_pytorch_lightning_example(modelstore: ModelStore) -> dict:
    # Load the data
    X_train, X_test, y_train, y_test = load_diabetes_dataset(as_numpy=True)

    data_set = TensorDataset(X_test, y_test)
    val_dataloader = DataLoader(data_set)

    data_set = TensorDataset(X_train, y_train)
    train_dataloader = DataLoader(data_set)

    # Train the model
    model = ExampleLightningNet()
    trainer = pl.Trainer(max_epochs=5, default_root_dir=mkdtemp())
    trainer.fit(model, train_dataloader, val_dataloader)

    # Upload the model to the model store
    print(
        f'⤴️  Uploading the pytorch lightning model to the "{_DIABETES_DOMAIN}" domain.'
    )
    return modelstore.upload(_DIABETES_DOMAIN, model=model, trainer=trainer)


def run_sklearn_example(modelstore: ModelStore) -> dict:
    # Load the data
    X_train, _, y_train, _ = load_diabetes_dataset()

    # Train a model using an sklearn pipeline
    params = {
        "n_estimators": 250,
        "max_depth": 4,
        "min_samples_split": 5,
        "learning_rate": 0.01,
        "loss": "ls",
    }
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("regressor", GradientBoostingRegressor(**params)),
        ]
    )
    pipeline.fit(X_train, y_train)

    # Upload the model to the model store
    print(
        f'⤴️  Uploading the sklearn pipeline to the "{_DIABETES_DOMAIN}" domain.'
    )
    return modelstore.upload(_DIABETES_DOMAIN, model=pipeline)


def run_tensorflow_example(modelstore: ModelStore) -> dict:
    # Load the data
    X_train, _, y_train, _ = load_diabetes_dataset()

    # Train a model
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Dense(5, activation="relu", input_shape=(10,)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mean_squared_error")
    model.fit(X_train, y_train, epochs=10)

    # Upload the model to the model store
    print(f'⤴️  Uploading the tf model to the "{_DIABETES_DOMAIN}" domain.')
    return modelstore.upload(_DIABETES_DOMAIN, model=model)


def run_transformers_example(modelstore: ModelStore) -> dict:
    model_name = "distilbert-base-cased"
    config = AutoConfig.from_pretrained(
        model_name,
        num_labels=2,
        finetuning_task="mnli",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        config=config,
    )

    # Skipped for brevity!
    # trainer = Trainer(
    #     model=model,
    #     args=training_args,
    #     train_dataset=train_dataset,
    #     eval_dataset=eval_dataset,
    #     compute_metrics=build_compute_metrics_fn(data_args.task_name),
    # )
    # trainer.train()

    # Upload the model to the model store
    model_domain = "example-distilbert-model"
    print(f'⤴️  Uploading the tf model to the "{model_domain}" domain.')
    return modelstore.upload(
        model_domain,
        config=config,
        model=model,
        tokenizer=tokenizer,
    )


def run_xgboost_example(modelstore: ModelStore) -> dict:
    # Load the data
    X_train, _, y_train, _ = load_diabetes_dataset()

    # Train a model
    xg_reg = xgb.XGBRegressor(
        objective="reg:squarederror",
        colsample_bytree=0.3,
        learning_rate=0.1,
        max_depth=5,
        alpha=10,
        n_estimators=10,
    )
    xg_reg.fit(X_train, y_train)

    # Upload the model to the model store
    print(
        f'⤴️  Uploading the xgboost model to the "{_DIABETES_DOMAIN}" domain.'
    )
    return modelstore.upload(_DIABETES_DOMAIN, model=model)
