# modelstore

`modelstore` is a Python library that allows you to version, export, save and download machine learning models in your choice of storage.

[![Downloads](https://pepy.tech/badge/modelstore)](https://pepy.tech/project/modelstore) [![Downloads](https://pepy.tech/badge/modelstore/month)](https://pepy.tech/project/modelstore)

💭  Give us feedback by completing this survey: https://forms.gle/XShU3zrZcnLRWsk36

[![](https://img.shields.io/static/v1?label=Sponsor&message=%E2%9D%A4&logo=GitHub&color=%23fe8e86)](https://github.com/sponsors/nlathia/)

## modelstore is an open source model registry

✅ No tracking server required
* Store models on a local file system or in a bucket
* Support for multiple clouds (AWS, GCP, Azure)

✅ Upload and version all your models
* Models are versioned on each upload
* Replaces all the boiler plate code you need to save models

✅ Manage models by domains and states
* List models in a domain
* Create model states and manage which state a model is in

✅ Download or load straight into memory
* Download models by id
* Load models straight from your storage back into memory

✅ Use as a command line tool
* Download models from the command line

For more details, please refer to [the documentation](https://modelstore.readthedocs.io/en/latest/).

## modelstore is being built in the open

💬 Come and find us in the [MLOps Community Slack](https://go.mlops.community/slack)'s `#oss-modelstore` channel.

## Installation

```python
pip install modelstore
```

## Supported storage types

* AWS S3 Bucket ([example](https://github.com/operatorai/modelstore/blob/b096275018674243835d21102f75b6270dfa2c97/examples/examples-by-storage/modelstores.py#L17-L21))
* Azure Blob Storage ([example](https://github.com/operatorai/modelstore/blob/b096275018674243835d21102f75b6270dfa2c97/examples/examples-by-storage/modelstores.py#L24-L31))
* Google Cloud Storage Bucket ([example](https://github.com/operatorai/modelstore/blob/b096275018674243835d21102f75b6270dfa2c97/examples/examples-by-storage/modelstores.py#L34-L41))
* Any s3-compatible object storage that you can access via [MinIO](https://min.io/)
* A filesystem directory ([example](https://github.com/operatorai/modelstore/blob/b096275018674243835d21102f75b6270dfa2c97/examples/examples-by-storage/modelstores.py#L44-L49))


## Supported machine learning libraries

* [Annoy](https://github.com/operatorai/modelstore/blob/main/examples/examples-by-ml-library/libraries/annoy_example.py)
* [Catboost](https://github.com/operatorai/modelstore/blob/main/examples/examples-by-ml-library/libraries/catboost_example.py)
* [Fast.AI](https://github.com/operatorai/modelstore/blob/main/examples/examples-by-ml-library/libraries/fastai_example.py)
* [Gensim](https://github.com/operatorai/modelstore/blob/main/examples/examples-by-ml-library/libraries/gensim_example.py)
* [Keras](https://github.com/operatorai/modelstore/blob/main/examples/examples-by-ml-library/libraries/keras_example.py)
* [LightGBM](https://github.com/operatorai/modelstore/blob/main/examples/examples-by-ml-library/libraries/lightgbm_example.py)
* [Mxnet](https://github.com/operatorai/modelstore/blob/main/examples/examples-by-ml-library/libraries/mxnet_example.py)
* [Onnx](https://github.com/operatorai/modelstore/blob/main/examples/examples-by-ml-library/libraries/onnx_sklearn_example.py)
* [Prophet](https://github.com/operatorai/modelstore/blob/main/examples/examples-by-ml-library/libraries/prophet_example.py)
* [PyTorch](https://github.com/operatorai/modelstore/blob/main/examples/examples-by-ml-library/libraries/pytorch_example.py)
* [PyTorch Lightning](https://github.com/operatorai/modelstore/blob/main/examples/examples-by-ml-library/libraries/pytorch_lightning_example.py)
* [Scikit Learn](https://github.com/operatorai/modelstore/blob/main/examples/examples-by-ml-library/libraries/sklearn_example.py)
* [Skorch](https://github.com/operatorai/modelstore/blob/main/examples/examples-by-ml-library/libraries/skorch_example.py)
* [Shap](https://github.com/operatorai/modelstore/blob/main/examples/examples-by-ml-library/libraries/shap_example.py)
* [Spark ML Lib](https://spark.apache.org/)
* [Tensorflow](https://github.com/operatorai/modelstore/blob/main/examples/examples-by-ml-library/libraries/tensorflow_example.py)
* Transformers - there are several examples in [this directory](https://github.com/operatorai/modelstore/tree/main/examples/examples-by-ml-library/libraries/huggingface)
* [XGBoost](https://github.com/operatorai/modelstore/blob/main/examples/examples-by-ml-library/libraries/xgboost_example.py)

Is there a machine learning framework that is missing? 
* Save your model and then upload it [as a raw file](https://github.com/operatorai/modelstore/blob/main/examples/examples-by-ml-library/libraries/raw_file_example.py).
* Feel free to [open an issue](https://github.com/operatorai/modelstore/issues)

## Read more about modelstore

* [Evidently.AI AMA with Neal Lathia](https://www.evidentlyai.com/blog/ama-neal-lathia), January 2023
* [MLOps Model Stores: Definition, Functionality, Tools Review](https://neptune.ai/blog/mlops-model-stores), January 2023
* [Monzo's machine learning stack](https://monzo.com/blog/2022/04/26/monzos-machine-learning-stack), April 2022
* [Data Talks Club Minis: Model Store](https://www.youtube.com/watch?v=85BWnKmOZl8), July 2021
* [Model arterfacts: the war stories](https://nlathia.github.io/2020/09/Model-artifacts-war-stories.html), September 2020

## Example Usage

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
meta_data = model_store.upload(domain, model=clf)

# Print the meta-data about the model
print(json.dumps(meta_data, indent=4))

# Load the model back!
clf = model_store.load(domain=model_domain, model_id=meta["model"]["model_id"])
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
