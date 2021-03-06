import json
import os

import click
from modelstore import ModelStore
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

import keras


def create_model_store(backend) -> ModelStore:
    if backend == "filesystem":
        # By default, create a new local model store
        # in our home directory
        home_dir = os.path.expanduser("~")
        return ModelStore.from_file_system(root_directory=home_dir)
    if backend == "gcloud":
        # The modelstore library assumes you have already created
        # a Cloud Storage bucket and will raise an exception if it doesn't exist
        return ModelStore.from_gcloud(
            os.environ["GCP_PROJECT_ID"],
            os.environ["GCP_BUCKET_NAME"],
        )
    if backend == "aws":
        # The modelstore library assumes that you already have
        # created an s3 bucket where you want to store your models, and
        # will raise an exception if it doesn't exist.
        return ModelStore.from_aws_s3(os.environ["AWS_BUCKET_NAME"])
    if backend == "hosted":
        # To use the hosted model store, you need an API key
        return ModelStore.from_api_key(
            os.environ["MODELSTORE_KEY_ID"], os.environ["MODELSTORE_ACCESS_KEY"]
        )
    raise ValueError(f"Unknown model store: {backend}")


def train():
    """
    Same model as
    https://keras.io/guides/serialization_and_saving/
    """
    diabetes = load_diabetes()
    X_train, X_test, y_train, y_test = train_test_split(
        diabetes.data, diabetes.target, test_size=0.1, random_state=13
    )
    inputs = keras.Input(shape=(10,))
    outputs = keras.layers.Dense(1)(inputs)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="mean_squared_error")

    model.fit(X_train, y_train, epochs=10)
    # Skipped for brevity (but important!) evaluate the model
    return model


@click.command()
@click.option(
    "--storage",
    type=click.Choice(
        ["filesystem", "gcloud", "aws", "hosted"], case_sensitive=False
    ),
)
def main(storage):
    model_type = "keras"
    model_domain = "diabetes-boosting-demo"

    # Create a model store instance
    model_store = create_model_store(storage)

    # In this demo, we train a single layer net
    # Replace this with the code to train your own model
    print(f"🤖  Training a {model_type} model...")
    model = train()

    # Upload the archive to the model store
    # The first string is the model's domain - which helps you to group
    # many models that are trained on the same target together
    print(f"⤴️  Uploading the archive to the {model_domain} domain.")
    meta_data = model_store.keras.upload(model_domain, model=model)

    # The upload returns meta-data about the model that was uploaded
    # This meta-data has also been sync'ed into the cloud storage
    #  bucket
    print(f"✅  Finished uploading the {model_type} model!")
    print(json.dumps(meta_data, indent=4))


if __name__ == "__main__":
    main()
