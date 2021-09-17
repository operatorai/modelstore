import json

import click

from libraries import annoy_example, catboost_example, fastai_example
from modelstores import create_model_store

EXAMPLES = {
    "annoy": annoy_example,
    "catboost": catboost_example,
    "fastai": fastai_example,
    # "gensim": run_gensim_example,
    # "keras": run_keras_example,
    # "lightgbm": run_lightgbm_example,
    # "file": run_model_file_example,
    # "pytorch": run_pytorch_example,
    # "pytorch-lightning": run_pytorch_lightning_example,
    # "sklearn": run_sklearn_example,
    # "tensorflow": run_tensorflow_example,
    # "transformers": run_transformers_example,
    # "xgboost": run_xgboost_example,
}


@click.command()
@click.option(
    "--modelstore-in",
    type=click.Choice(
        ["aws", "azure", "gcloud", "filesystem", "hosted"], case_sensitive=False
    ),
)
@click.option(
    "--ml-framework",
    type=click.Choice(
        EXAMPLES.keys(),
        case_sensitive=False,
    ),
)
def main(modelstore_in, ml_framework):
    print(
        f"\n🆕  Running {ml_framework} modelstore example with {modelstore_in} backend."
    )

    # Create a model store instance
    modelstore = create_model_store(modelstore_in)

    # Run the example: train and upload a model
    example = EXAMPLES[ml_framework]
    meta_data = example.train_and_upload(modelstore)

    # Run the example: download and load a model
    model_id = meta_data["model"]["model_id"]
    example.load_and_test(modelstore, model_id)

    # The upload returns meta-data about the model that was uploaded
    # In this example, we just print it out to the terminal
    print(f"✅  Finished uploading the {ml_framework} model!")
    print(json.dumps(meta_data, indent=4))


if __name__ == "__main__":
    main()
