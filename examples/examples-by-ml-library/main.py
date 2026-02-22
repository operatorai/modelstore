#    Copyright 2023 Neal Lathia
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
import importlib
import sys

import click
from modelstores import MODELSTORES, create_model_store

EXAMPLES = {
    "annoy": "libraries.annoy_example",
    "catboost": "libraries.catboost_example",
    "causalml": "libraries.causalml_example",
    "dpt": "libraries.huggingface.dpt",
    "fastai": "libraries.fastai_example",
    "file": "libraries.raw_file_example",
    "gensim": "libraries.gensim_example",
    "hf-distilbert": "libraries.huggingface.distilbert",
    "hf-gpt2-pt": "libraries.huggingface.gpt2_pytorch",
    "hf-gpt2-tf": "libraries.huggingface.gpt2_tensorflow",
    "keras": "libraries.keras_example",
    "lightgbm": "libraries.lightgbm_example",
    "onnx-sklearn": "libraries.onnx_sklearn_example",
    "onnx-lightgbm": "libraries.onnx_lightgbm_example",
    "prophet": "libraries.prophet_example",
    "pyspark": "libraries.pyspark_example",
    "pytorch": "libraries.pytorch_example",
    "pytorch-lightning": "libraries.pytorch_lightning_example",
    "segment-anything": "libraries.huggingface.sam",
    "shap": "libraries.shap_example",
    "sklearn": "libraries.sklearn_example",
    "sklearn-with-explainer": "libraries.sklearn_with_explainer_example",
    "sklearn-with-extras": "libraries.sklearn_with_extras_example",
    "skorch": "libraries.skorch_example",
    "tensorflow": "libraries.tensorflow_example",
    "xgboost": "libraries.xgboost_example",
    "xgboost-booster": "libraries.xgboost_booster_example",
    "yolov5": "libraries.yolo_example",
}


@click.command()
@click.option(
    "--modelstore-in",
    type=click.Choice(
        MODELSTORES.keys(),
        case_sensitive=False,
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
    if sys.platform == "darwin" and ml_framework in ["fastai", "pyspark"]:
        print(f"⏩  Skipping {ml_framework} on darwin.")
        return
    if ml_framework == "yolov5" and modelstore_in == "azure-container":
        # Upload time out bug
        print(f"⏩  Skipping {ml_framework} in {modelstore_in}.")
        return
    print(
        f"\n🆕  Running {ml_framework} modelstore example with {modelstore_in} backend."
    )

    # Create a model store instance
    modelstore = create_model_store(modelstore_in)
    example = importlib.import_module(EXAMPLES[ml_framework])

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
