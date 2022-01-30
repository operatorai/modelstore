import json

import click

from libraries import (
    annoy_example,
    catboost_example,
    fastai_example,
    gensim_example,
    keras_example,
    lightgbm_example,
    mxnet_example,
    onnx_example,
    prophet_example,
    pytorch_example,
    pytorch_lightning_example,
    raw_file_example,
    shap_example,
    sklearn_example,
    sklearn_with_explainer_example,
    skorch_example,
    tensorflow_example,
    transformers_example,
    xgboost_example,
)
from modelstores import create_model_store

EXAMPLES = {
    "annoy": annoy_example,
    "catboost": catboost_example,
    "fastai": fastai_example,
    "file": raw_file_example,
    "gensim": gensim_example,
    "keras": keras_example,
    "lightgbm": lightgbm_example,
    "mxnet": mxnet_example,
    "onnx": onnx_example,
    "prophet": prophet_example,
    "pytorch": pytorch_example,
    "pytorch-lightning": pytorch_lightning_example,
    "shap": shap_example,
    "sklearn": sklearn_example,
    "sklearn-with-explainer": sklearn_with_explainer_example,
    "skorch": skorch_example,
    "tensorflow": tensorflow_example,
    "transformers": transformers_example,
    "xgboost": xgboost_example,
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

    if modelstore_in != "hosted":
        # Run the example: downloading and loading a model
        # is currently unimplemented in the "hosted" storage
        model_id = meta_data["model"]["model_id"]
        example.load_and_test(modelstore, model_id)

    # The upload returns meta-data about the model that was uploaded
    # In this example, we just print it out to the terminal
    print(f"✅  Finished uploading the {ml_framework} model!")
    print(json.dumps(meta_data, indent=4))


if __name__ == "__main__":
    main()
