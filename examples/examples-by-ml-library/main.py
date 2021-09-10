import json

import click

from models import (
    run_annoy_example,
    run_catboost_example,
    run_fastai_example,
    run_gensim_example,
    run_keras_example,
    run_lightgbm_example,
    run_model_file_example,
    run_pytorch_example,
    run_pytorch_lightning_example,
    run_sklearn_example,
    run_tensorflow_example,
    run_transformers_example,
    run_xgboost_example,
)
from modelstores import create_model_store

EXAMPLES = {
    "annoy": run_annoy_example,
    "catboost": run_catboost_example,
    "fastai": run_fastai_example,
    "gensim": run_gensim_example,
    "keras": run_keras_example,
    "lightgbm": run_lightgbm_example,
    "file": run_model_file_example,
    "pytorch": run_pytorch_example,
    "pytorch-lightning": run_pytorch_lightning_example,
    "sklearn": run_sklearn_example,
    "tensorflow": run_tensorflow_example,
    "transformers": run_transformers_example,
    "xgboost": run_xgboost_example,
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

    # Run the example
    example_function = EXAMPLES[ml_framework]
    meta_data = example_function(modelstore)

    # The upload returns meta-data about the model that was uploaded
    # In this example, we just print it out to the terminal
    print(f"✅  Finished uploading the {ml_framework} model!")
    print(json.dumps(meta_data, indent=4))


if __name__ == "__main__":
    main()
