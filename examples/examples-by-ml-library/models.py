import tempfile

import xgboost as xgb
from modelstore.model_store import ModelStore
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from networks import ExampleLightningNet, ExampleNet

# pylint: disable=invalid-name


def run_model_file_example(modelstore: ModelStore) -> dict:
    # For this demo, we use a temporary directory. This is not required
    #  in your code -- just point modelstore directly to the model file
    # that you want to persist
    with tempfile.TemporaryDirectory() as tmp_dir:

        # Create a file with a "model" -- in this case it is a json file,
        # but modelstore can handle any file type
        model_path = os.path.join(tmp_dir, "model.json")
        with open(model_path, "w") as out:
            out.write(json.dumps({"weights": [0.1, 0.2, 0.3]}))

        model_domain = "example-model-file"
        print(f'⤴️  Uploading the model file to the "{model_domain}" domain.')
        return modelstore.upload(model_domain, model=model_path)


def run_sklearn_example(modelstore: ModelStore) -> dict:
    # Load the data
    X_train, X_test, y_train, y_test = load_diabetes_dataset()

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
    results = mean_squared_error(y_test, pipeline.predict(X_test))
    print(f"🔍  Trained model MSE={results}.")

    # Upload the model to the model store
    print(
        f'⤴️  Uploading the sklearn pipeline to the "{_DIABETES_DOMAIN}" domain.'
    )
    meta_data = modelstore.upload(_DIABETES_DOMAIN, model=pipeline)

    # Load the model back into memory!
    model_id = meta_data["model"]["model_id"]
    print(
        f'⤵️  Loading the sklearn "{_DIABETES_DOMAIN}" domain model={model_id}'
    )
    model = modelstore.load(_DIABETES_DOMAIN, model_id)

    results = mean_squared_error(y_test, model.predict(X_test))
    print(f"🔍  Loaded model MSE={results}.")

    return meta_data


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

    results = mean_squared_error(y_test, model.predict(X_test))
    print(f"🔍  Trained model MSE={results}.")

    # Upload the model to the model store
    meta_data = modelstore.upload(_DIABETES_DOMAIN, model=model)

    # Load the model back into memory!
    model_id = meta_data["model"]["model_id"]
    print(
        f'⤵️  Loading the tensorflow "{_DIABETES_DOMAIN}" domain model={model_id}'
    )
    model = modelstore.load(_DIABETES_DOMAIN, model_id)
    results = mean_squared_error(y_test, model.predict(X_test))
    print(f"🔍  Loaded model MSE={results}.")

    return meta_data


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
    print(
        f'⤴️  Uploading the transformers model to the "{model_domain}" domain.'
    )
    meta_data = modelstore.upload(
        model_domain,
        config=config,
        model=model,
        tokenizer=tokenizer,
    )

    # Load the model back into memory!
    model_id = meta_data["model"]["model_id"]
    print(
        f'⤵️  Loading the transformers "{model_domain}" domain model={model_id}'
    )
    model, tokenizer, config = modelstore.load(model_domain, model_id)

    return meta_data


def run_xgboost_example(modelstore: ModelStore) -> dict:
    # Load the data
    X_train, X_test, y_train, y_test = load_diabetes_dataset()

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

    results = mean_squared_error(y_test, xg_reg.predict(X_test))
    print(f"🔍  Trained model MSE={results}.")

    # Upload the model to the model store
    print(
        f'⤴️  Uploading the xgboost model to the "{_DIABETES_DOMAIN}" domain.'
    )
    # Alternative: modelstore.catboost.upload(model_domain, model=model)
    meta_data = modelstore.upload(_DIABETES_DOMAIN, model=xg_reg)

    # Load the model back into memory!
    model_id = meta_data["model"]["model_id"]
    print(
        f'⤵️  Loading the xgboost "{_DIABETES_DOMAIN}" domain model={model_id}'
    )
    xg_reg = modelstore.load(_DIABETES_DOMAIN, model_id)

    results = mean_squared_error(y_test, xg_reg.predict(X_test))
    print(f"🔍  Loaded model MSE={results}.")

    return meta_data
