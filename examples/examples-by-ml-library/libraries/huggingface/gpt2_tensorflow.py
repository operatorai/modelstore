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

from transformers import GPT2Tokenizer, TFGPT2LMHeadModel
from modelstore.model_store import ModelStore


_DOMAIN_NAME = "example-gpt2-model"


def _run_prediction(model: TFGPT2LMHeadModel, tokenizer: GPT2Tokenizer):
    text = "What is MLOps, and why is it important?"
    encoded_input = tokenizer(text, return_tensors="tf")
    output = model.generate(**encoded_input)
    decoded = tokenizer.decode(output[0])
    print(f"🔍 Model output={decoded}.")


def _train_example_model():
    # Returns a Tensorflow model
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = TFGPT2LMHeadModel.from_pretrained("gpt2")

    _run_prediction(model, tokenizer)
    return model, tokenizer


def train_and_upload(modelstore: ModelStore) -> dict:
    # Train a model
    model, tokenizer = _train_example_model()

    # Upload the model to the model store
    print(f'⤴️  Uploading the transformers model to the "{_DOMAIN_NAME}" domain.')
    meta_data = modelstore.upload(
        _DOMAIN_NAME,
        model=model,
        tokenizer=tokenizer,
    )
    return meta_data


def load_and_test(modelstore: ModelStore, model_domain: str, model_id: str):
    # Load the model back into memory!
    print(f'⤵️  Loading the transformers "{model_domain}" domain model={model_id}')
    model, tokenizer, _ = modelstore.load(model_domain, model_id)
    _run_prediction(model, tokenizer)
