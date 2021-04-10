import json
import os

import click
from modelstore import ModelStore
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from gensim.models import word2vec


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


@click.command()
@click.option(
    "--storage",
    type=click.Choice(
        ["filesystem", "gcloud", "aws", "hosted"], case_sensitive=False
    ),
)
def main(storage):
    model_domain = "newsgroups-embeddings"

    # Create a model store instance
    model_store = create_model_store(storage)

    # Load the data
    categories = [
        "alt.atheism",
        "soc.religion.christian",
        "comp.graphics",
        "sci.med",
    ]
    print(f"⏳  Fetching the newsgroups data...")
    newsgroups = fetch_20newsgroups(
        subset="train", categories=categories, shuffle=True, random_state=42
    )
    sentences = [doc.strip().split() for doc in newsgroups.data]

    # Train a word2vec model
    print(f"🤖  Training a word2vec model...")
    model = word2vec.Word2Vec(sentences, min_count=2)

    print(f"⤴️  Uploading the model to the {model_domain} domain.")
    meta_data = model_store.gensim.upload(model_domain, model=model)

    # The upload returns meta-data about the model that was uploaded
    print(f"✅  Finished uploading the gensim model!")
    print(json.dumps(meta_data, indent=4))


if __name__ == "__main__":
    main()
