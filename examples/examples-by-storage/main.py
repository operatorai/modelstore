import json
import os

import click
import numpy as np
import xgboost as xgb
from sklearn.datasets import load_diabetes
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from modelstores import create_model_store


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
    "--modelstore-in",
    type=click.Choice(
        ["aws", "azure", "gcloud", "filesystem", "hosted"], case_sensitive=False
    ),
)
def main(modelstore_in):
    print(f"\n🆕  Running modelstore example with {modelstore_in} backend.")

    # Create a model store instance
    modelstore = create_model_store(modelstore_in)
    model_domain = "diabetes-boosting-demo"

    # In this demo, we train two models, so that we can demonstrate
    # how modelstore keeps track of uploaded models for us
    for model_type in ["sklearn", "xgboost"]:
        print(f"🤖  Training a {model_type} model...")
        model = train(model_type)

        print(f"⤴️  Uploading to the {model_domain} domain.")
        meta_data = modelstore.upload(model_domain, model=model)

        # The upload returns meta-data about the model that was uploaded
        # This meta-data has also been sync'ed into the s3 bucket
        print(f"✅  Finished uploading the {model_type} model!")
        print(json.dumps(meta_data, indent=4))

        # Download the model back!
        target = os.path.join(modelstore_in, f"downloaded-{model_type}-model")
        os.makedirs(target, exist_ok=True)
        model_path = modelstore.download(
            local_path=target,
            domain=model_domain,
            model_id=meta_data["model"]["model_id"],
        )
        print(f"⤵️  Downloaded the model back to {model_path}")

    # List the available domains
    print(f"✅  Listing existing domains:")
    domains = modelstore.list_domains()
    for domain in domains:
        print(f"\t  Domain: {domain}")

    # List the available models in the diabest-boosting-demo domain
    print(f"✅  Listing models for {model_domain}:")
    versions = modelstore.list_versions(domain=model_domain)
    for version in versions:
        print(f"\t  Domain: {model_domain} has model with id={version}")

    if modelstore_in == "hosted":
        # The rest is currently not implemented in the 'hosted'
        # modelstore
        return

    # Create a new model state
    state_prod = "production"
    print(f"✅  Creating model state={state_prod}:")
    modelstore.create_model_state(state_prod)

    # Set the first model to the production state
    print(f"✅  Setting model_id={versions[0]} to state={state_prod}:")
    modelstore.set_model_state(model_domain, versions[0], state_prod)

    # List the models that are in production
    print(
        f"✅  Listing models for {model_domain} that are in state={state_prod}:"
    )
    versions = modelstore.list_versions(model_domain, state_name=state_prod)
    for version in versions:
        print(f"\t  Domain: {model_domain} has model with id={version}")


if __name__ == "__main__":
    main()
