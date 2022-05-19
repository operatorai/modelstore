import click

from libraries import (
    annoy_example,
    catboost_example,
    fastai_example,
    gensim_example,
    keras_example,
    lightgbm_example,
    mxnet_example,
    onnx_sklearn_example,
    onnx_lightgbm_example,
    prophet_example,
    pytorch_example,
    pytorch_lightning_example,
    raw_file_example,
    shap_example,
    sklearn_example,
    sklearn_with_explainer_example,
    sklearn_with_extras_example,
    skorch_example,
    tensorflow_example,
    transformers_example,
    xgboost_booster_example,
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
    "onnx-sklearn": onnx_sklearn_example,
    "onnx-lightgbm": onnx_lightgbm_example,
    "prophet": prophet_example,
    "pytorch": pytorch_example,
    "pytorch-lightning": pytorch_lightning_example,
    "shap": shap_example,
    "sklearn": sklearn_example,
    "sklearn-with-explainer": sklearn_with_explainer_example,
    "sklearn-with-extras": sklearn_with_extras_example,
    "skorch": skorch_example,
    "tensorflow": tensorflow_example,
    "transformers": transformers_example,
    "xgboost": xgboost_example,
    "xgboost-booster": xgboost_booster_example,
}


@click.command()
@click.option(
    "--modelstore-in",
    type=click.Choice(["aws", "azure", "gcloud", "filesystem"], case_sensitive=False),
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
    example = EXAMPLES[ml_framework]

    # Demo how we train and upload a model
    meta_data = example.train_and_upload(modelstore)
    model_domain = meta_data["model"]["domain"]
    model_id = meta_data["model"]["model_id"]
    model_type = meta_data["model"]["model_type"]["library"]

    print(f"✅  Finished uploading the {ml_framework} model! (detected: {model_type})")

    # Demo how we can load the model back
    example.load_and_test(modelstore, model_domain, model_id)
    print(f"✅  Finished loading the {ml_framework} model!")

    # Since this is a demo-only, the model is deleted
    modelstore.delete_model(model_domain, model_id, skip_prompt=True)
    print(f"✅  The {ml_framework} model has been deleted!")


if __name__ == "__main__":
    main()
