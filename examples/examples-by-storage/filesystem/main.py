import json
import os

import click
import numpy as np
import xgboost as xgb
from modelstore import ModelStore
from sklearn.datasets import load_diabetes
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


def create_model_store() -> ModelStore:
    # By default, create a new local model store
    # in our home directory
    home_dir = os.path.expanduser("~")
    return ModelStore.from_file_system(root_directory=home_dir)


def train(model_type):
    diabetes = load_diabetes()
    X_train, X_test, y_train, y_test = train_test_split(
        diabetes.data, diabetes.target, test_size=0.1, random_state=13
    )

    if model_type == "sklearn":
        params = {
            "n_estimators": 500,
            "max_depth": 4,
            "min_samples_split": 5,
            "learning_rate": 0.01,
            "loss": "ls",
        }
        reg = GradientBoostingRegressor(**params)
        reg.fit(X_train, y_train)
        # Skipped for brevity (but important!) evaluate the model
        return reg
    if model_type == "xgboost":
        xg_reg = xgb.XGBRegressor(
            objective="reg:squarederror",
            colsample_bytree=0.3,
            learning_rate=0.1,
            max_depth=5,
            alpha=10,
            n_estimators=10,
        )
        xg_reg.fit(X_train, y_train)

        preds = xg_reg.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        print(f"📈  Trained a model with RMSE={rmse}.")
        return xg_reg


@click.command()
@click.option(
    "--model-type",
    type=click.Choice(["sklearn", "xgboost"], case_sensitive=False),
)
def main(model_type):
    model_domain = "diabetes-boosting-demo"

    # Create a model store instance
    model_store = create_model_store()

    # In this demo, we train a GradientBoostingRegressor
    # using the same approach described on the scikit-learn website.
    # Replace this with the code to train your own model
    print(f"🤖  Training a {model_type} model...")
    model = train(model_type)

    # Create an archive containing the trained model
    print(f"⤴️  Uploading the archive to the {model_domain} domain.")
    if model_type == "sklearn":
        meta_data = model_store.sklearn.upload(model_domain, model=model)
    elif model_type == "xgboost":
        meta_data = model_store.sklearn.upload(model_domain, model=model)
    else:
        raise NotImplementedError(f"Not implemented for: {model_type}")

    # The upload returns meta-data about the model that was uploaded
    # This meta-data has also been sync'ed into the file system
    print(f"✅  Finished uploading the {model_type} model!")
    print(json.dumps(meta_data, indent=4))

    # Download the model back!
    target = f"downloaded-{model_type}-model"
    os.makedirs(target, exist_ok=True)
    model_path = model_store.download(
        local_path=target,
        domain=model_domain,
        model_id=meta_data["model"]["model_id"],
    )
    print(f"⤵️  Downloaded the model back to {model_path}")


if __name__ == "__main__":
    main()
