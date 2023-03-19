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


def load_and_test(modelstore: ModelStore, model_domain: str, model_id: str):
    # Loading the model back into memory is not supported
    # for models that have been saved to disk manually
    pass
