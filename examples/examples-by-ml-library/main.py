import json

import click

from catboost_example import run_catboost_example
from modelstores import create_model_store

EXAMPLES = {
    "catboost": run_catboost_example,
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
        [
            "catboost",
            "fastai",
            "gensim",
            "keras",
            "lightgbm",
            "pytorch",
            "pytorch-lightning",
            "sklearn",
            "tensorflow",
            "transformers",
            "xgboost",
        ],
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
