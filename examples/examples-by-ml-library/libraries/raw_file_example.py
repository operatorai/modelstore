import json
import os
import tempfile

from modelstore.model_store import ModelStore

_DOMAIN_NAME = "example-model-file"


def _train_and_save_example_model(tmp_dir: str) -> str:
    # Create a file with a "model" -- in this case it is a json file,
    # but modelstore can handle any file type
    model_path = os.path.join(tmp_dir, "model.json")
    with open(model_path, "w") as out:
        out.write(json.dumps({"weights": [0.1, 0.2, 0.3]}))
    return model_path


def train_and_upload(modelstore: ModelStore) -> dict:
    # Train a "model" and save it into a temp directory
    with tempfile.TemporaryDirectory() as tmp_dir:
        model_path = _train_and_save_example_model(tmp_dir)

        # Upload the model to the model store
        print(f'⤴️  Uploading the saved model to the "{_DOMAIN_NAME}" domain.')
        meta_data = modelstore.upload(_DOMAIN_NAME, model=model_path)
    return meta_data


def load_and_test(modelstore: ModelStore, model_id: str):
    # Loading the model back into memory is not supported
    # for models that have been saved to disk manually
    pass
