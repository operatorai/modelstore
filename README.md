# modelstore

`modelstore` is a Python library that allows you to version, export, save and download machine learning models.

The library's `ModelStore` will (a) version your models, (b) store them in a structured way, and (c) collect meta-data about the Python runtime that was used to train them.

For more details, please refer to [the documentation](https://modelstore.readthedocs.io/en/latest/).

This library has been developed using Python `3.6` and `3.7`. Please open an issue if you have any trouble!


## Installation

```python
pip install modelstore
```

[![Downloads](https://pepy.tech/badge/modelstore)](https://pepy.tech/project/modelstore)

## Supported storage types

* AWS S3 Bucket ([example](https://github.com/operatorai/modelstore/blob/master/examples/examples-by-storage/aws/main.py))
* Google Cloud Storage Bucket ([example](https://github.com/operatorai/modelstore/blob/master/examples/examples-by-storage/gcloud/main.py))
* A filesystem directory ([example](https://github.com/operatorai/modelstore/blob/master/examples/examples-by-storage/filesystem/main.py))
* A hosted storage option (Get in touch for an API key! [example](https://github.com/operatorai/modelstore/blob/master/examples/examples-by-storage/hosted/main.py))

## Supported machine learning libraries

* [Scikit Learn](https://github.com/operatorai/modelstore/blob/master/examples/examples-by-ml-library/sklearn/main.py)
* [Catboost](https://github.com/operatorai/modelstore/blob/master/examples/examples-by-ml-library/catboost/main.py#L65-L70)
* [Keras](https://github.com/operatorai/modelstore/blob/master/examples/examples-by-ml-library/keras/main.py)
* [Tensorflow](https://github.com/operatorai/modelstore/blob/master/examples/examples-by-ml-library/tensorflow/main.py)
* [LightGBM](https://github.com/operatorai/modelstore/blob/master/examples/examples-by-ml-library/lightgbm/main.py)
* [PyTorch](https://github.com/operatorai/modelstore/blob/master/examples/examples-by-ml-library/pytorch/main.py)
* [PyTorch Lightning](https://github.com/operatorai/modelstore/tree/master/examples/examples-by-ml-library/pytorch-lightning/main.py)
* [Transformers](https://github.com/operatorai/modelstore/blob/master/examples/examples-by-ml-library/transformers/main.py)
* [XGBoost](https://github.com/operatorai/modelstore/blob/master/examples/examples-by-ml-library/xgboost/main.py)

Is there a machine learning framework that is missing? Feel free to [open an issue](https://github.com/operatorai/modelstore/issues) or send us a pull request!

## Usage

### Colab Notebook

There is a [full example in this Colab notebook](https://colab.research.google.com/drive/1yEY6wy68k7TlHzm8iJMKKBG_Pl-MGZUe?usp=sharing).

### Python Script

```python
from modelstore import ModelStore
# And your other imports

# Train your model
clf = RandomForestClassifier(n_estimators=10)
clf = clf.fit(X, Y)

# Create a model store that uses a one of the storage options
# In this example, the model store is created with a GCP bucket
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
