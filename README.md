# modelstore

`modelstore` is a Python library that allows you to export, save and version machine learning models.

The library's `ModelStore` will (a) version your models, (b) store them in a structured way, and (c) collect meta-data about the Python runtime that was used to train them.

This library has been developed using Python `3.6` and `3.7` and is in pre-alpha. Please open an issue if you have any trouble!

## Installation

```python
pip install modelstore
```

## Usage

```python
from modelstore import ModelStore
# And your other imports

# Train your model
clf = RandomForestClassifier(n_estimators=10)
clf = clf.fit(X, Y)

# Create a model store that uses a Google Cloud bucket (or AWS bucket)
model_store = ModelStore.from_gcloud(
   project_name="my-project",
   bucket_name="my-bucket",
)

# Upload the archive to your model store
domain = "example-model"
meta_data = model_store.sklearn.upload(domain, model=clf)

# Print the meta-data about the model
print(json.dumps(meta_data, indent=4))

# Download the model back!
target = f"downloaded-{model_type}-model"
os.makedirs(target, exist_ok=True)
model_path = model_store.download(
   local_path=target,
   domain=model_domain,
   model_id=meta["model"]["model_id"],
)
```

For more details, please refer to [the documentation](https://modelstore.readthedocs.io/en/latest/).

## License

Copyright 2020 Neal Lathia

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
