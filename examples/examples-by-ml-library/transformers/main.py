import json
import os

import click
from modelstore import ModelStore

from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


def create_model_store(backend) -> ModelStore:
    if backend == "filesystem":
        # By default, create a new local model store one directory up
        #  from the current example that is being run
        return ModelStore.from_file_system(root_directory="~")
    if backend == "gcloud":
        # The modelstore library assumes you have already created
        # a Cloud Storage bucket and will raise an exception if it doesn't exist
        return ModelStore.from_gcloud(
            os.environ["GCP_PROJECT_ID"], os.environ["GCP_BUCKET_NAME"],
        )
    if backend == "aws":
        # The modelstore library assumes that you already have
        # created an s3 bucket where you want to store your models, and
        # will raise an exception if it doesn't exist.
        return ModelStore.from_aws_s3(os.environ["AWS_BUCKET_NAME"])
    else:
        raise ValueError(f"Unknown model store: {backend}")


def train():
    model_name = "distilbert-base-cased"
    config = AutoConfig.from_pretrained(
        model_name, num_labels=2, finetuning_task="mnli",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, config=config,
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
    return config, model, tokenizer


@click.command()
@click.option(
    "--storage",
    type=click.Choice(["filesystem", "gcloud", "aws"], case_sensitive=False),
)
def main(storage):
    model_type = "transformers"
    model_domain = "example-distilbert-model"

    # Create a model store instance
    model_store = create_model_store(storage)

    # In this demo, we train a single layered net
    # using the sklearn.datasets.load_diabetes dataset
    print(f"🤖  Creating a model using {model_type}...")
    config, model, tokenizer = train()

    # Upload the archive to the model store
    # Model domains help you to group many models together
    print(f"⤴️  Uploading the archive to the {storage} {model_domain} domain.")
    meta_data = model_store.transformers.upload(
        model_domain, config=config, model=model, tokenizer=tokenizer,
    )

    # The upload returns meta-data about the model that was uploaded
    # This meta-data has also been sync'ed into the cloud storage
    #  bucket
    print(f"✅  Finished uploading the {model_type} model!")
    print(json.dumps(meta_data, indent=4))


if __name__ == "__main__":
    main()
