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
from typing import Tuple
from transformers import (
    DPTImageProcessor,
    DPTPreTrainedModel,
    DPTForDepthEstimation
)
from modelstore.model_store import ModelStore


_DOMAIN_NAME = "example-dpt-model"


def _load_dpt_model(
        source: str = "Intel/dpt-large") -> Tuple[DPTPreTrainedModel, DPTImageProcessor]:
    print(f"Loading a dpt anything model from:{source}.")
    processor = DPTImageProcessor.from_pretrained(source)
    model = DPTForDepthEstimation.from_pretrained(source)
    return model, processor


def train_and_upload(modelstore: ModelStore) -> dict:
    # Train a model
    model, processor = _load_dpt_model()

    # Upload the model to the model store
    print(f'⤴️  Uploading the transformers model to the "{_DOMAIN_NAME}" domain.')
    meta_data = modelstore.upload(
        _DOMAIN_NAME,
        model=model,
        processor=processor,
    )
    return meta_data


def load_and_test(modelstore: ModelStore, model_domain: str, model_id: str):
    # Load the model back into memory!
    print(f'⤵️  Loading the transformers "{model_domain}" domain model={model_id}')
    model, processor, config = modelstore.load(model_domain, model_id)
    
    print(f"Loaded model={type(model)}")
    print(f"Loaded processor={type(processor)}")
    print(f"Loaded config={type(config)}")
    # Run some example predictions
    # ...
