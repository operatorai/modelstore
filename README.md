# modelstore

`modelstore` is a Python library that allows you to version, export, and save a machine learning models
to your filesystem or a cloud storage provider (AWS or GCP).

The library's `ModelStore` helps you to (a) version your mdoels, (b) store them in a structured way, and
(c) collect meta data about the Python runtime that was used to train them.

This library has been developed using Python `3.7.0` and is in pre-alpha. Please open an issue if you have
any trouble!

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
ms = ModelStore.from_gcloud(
   project_name="my-project",
   bucket_name="my-bucket",
)

# Create an archive that exports your model
archive = ms.sklearn.create_archive(model=clf)

# Upload the archive to your model store
meta_data = ms.upload(domain="example-model", archive)

print(json.dumps(meta_data, indent=4))
```

For more details, please refer to the documentation.


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
