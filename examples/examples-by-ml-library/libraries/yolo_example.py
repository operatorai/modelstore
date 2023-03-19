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

import torch
from modelstore.model_store import ModelStore

_YOLO_DOMAIN = "yolov5"


def _predict(model):
    model.eval()
    img = "https://ultralytics.com/images/zidane.jpg"
    results = model(img)
    print(f"🔍  Prediction result: \n{results.pandas().xyxy[0]}.")


def train_and_upload(modelstore: ModelStore) -> dict:
    # Load the yolov5 model
    model = torch.hub.load("ultralytics/yolov5", "yolov5s")
    _predict(model)

    # Upload the model to the model store
    print(f'⤴️  Uploading the yolo model to the "{_YOLO_DOMAIN}" domain.')
    meta_data = modelstore.upload(_YOLO_DOMAIN, model=model)
    return meta_data


def load_and_test(modelstore: ModelStore, model_domain: str, model_id: str):
    # Load the model back into memory!
    print(f'⤵️  Loading the yolo "{model_domain}" domain model={model_id}')
    model = modelstore.load(model_domain, model_id)
    _predict(model)
