# Change log

## modelstore 0.0.74 (April 2022)

**🆕  New functionality**

`get_domain()` returns key meta data about a domain ([#141](https://github.com/operatorai/modelstore/pull/141))

`delete_model()` delete models from `modelstore`. If the user attempts to query for a model after it has been deleted, `modelstore` will raise a `ModelDeletedException` ([#137](https://github.com/operatorai/modelstore/pull/137))

`list_model_states()` lists all of the existing model states ([#131](https://github.com/operatorai/modelstore/pull/131))

You can optionally set a `model_id` value when uploading a model ([#147](https://github.com/operatorai/modelstore/pull/147/), [#165](https://github.com/operatorai/modelstore/pull/165)), thanks [@cdknorow](https://github.com/cdknorow). 

**🆕  Storage improvements**

The file system storage can now be configured to create its root directory if it doesn't already exist ([#143](https://github.com/operatorai/modelstore/pull/143/), thanks [@cdknorow](https://github.com/cdknorow))

Public, read-only Google Cloud Storage containers can now be read from using `modelstore` ([#142](https://github.com/operatorai/modelstore/pull/142), thanks [@ionicsolutions](https://github.com/ionicsolutions))

Previously, any extra files you wanted to upload were uploaded separately to the model archive. Now, they are added into the archive in a subdirectory called "extras" so that you can easily download them back ([#139](https://github.com/operatorai/modelstore/pull/139)). I've also added an example of uploading a model with some additional files ([#138](https://github.com/operatorai/modelstore/pull/138)). 

**🐛  Bug fixes & general updates**

Fixed a regression: `keras` models saved with an older version of `modelstore` couldn't be loaded ([#145](https://github.com/operatorai/modelstore/pull/145)).

Updated the names of the environment variables that are checked for setting the modelstore storage root (prefixes). Previously, this was using the same variable name and this would cause issues if you were creating more than one type of modelstore.

**ℹ️  General updates**

The `list_versions()` function is deprecated and has been replaced with `list_models()` ([#132](https://github.com/operatorai/modelstore/pull/132))

Python 3.6 has passed its end-of-life, so this library is now tested with Python 3.7 and above.

## modelstore 0.0.73

🆕   You can upload multiple models to the same archive, if they don't share any keywords. For example `modelstore.upload(domain, model=sklearn_model, explainer=shap_explainer)` can be used to upload and download models and explainers together.

🐛   Merged the model managers for `keras` and `tensorflow` into one.

🆕   You can now set the root prefix of your model registry storage (thank you, [@cdknorow](https://github.com/cdknorow)!).

🆕   Added to the CLI functionality! You can now `python -m modelstore upload <domain> <model-file>` to upload a model. This requires you to [set environment variables](https://modelstore.readthedocs.io/en/latest/concepts/cli.html).

🆕   Added support for uploading [skorch](https://github.com/skorch-dev/skorch) models

## modelstore 0.0.72

🐛  Saving complex sklearn pipelines was [raising a TypeError](https://github.com/operatorai/modelstore/issues/96). This is because the `get_params()` function, which `modelstore` uses to save meta data about the model, returns a lot of things that are not JSON serializable. For now, I've patched this by not returning metadata for `sklearn.pipeline.Pipeline` models.

🐛  Colab is currently running `fastai==1.0.61`, while `modelstore` was designed for `fastai>2`, so things would [break in Colab Notebooks](https://github.com/operatorai/modelstore/issues/95) due to the different import paths in the two versions of `fastai`: the import paths are now version-dependent..

🐛  Updated the library so that PyTorch models can be uploaded [without an optimizer](https://github.com/operatorai/modelstore/issues/94). This is useful for uploading pretrained embedding models!

🐛  Fixed [a logging bug when trying to download the latest model](https://github.com/operatorai/modelstore/issues/91) in a domain - hat tip to [@erosenthal-square](https://github.com/erosenthal-square) who found the issue.

🆕   Added support for uploading [unsetting model states](https://github.com/operatorai/modelstore/issues/82) - hat tip to [@erosenthal-square](https://github.com/erosenthal-square) who opened an issue about this.

🆕   Added support for uploading [shap](https://shap.readthedocs.io/en/latest/generated/shap.Explainer.html) explainers.

🐛  Fixed an `ImportError` bug when trying to use `modelstore` on an instance [that does not have git installed](https://github.com/operatorai/modelstore/pull/86).

🆕   Added CLI functionality! You can now `python -m modelstore download <domain> <model-id> <directory>` to download a model. This requires you to [set environment variables](https://modelstore.readthedocs.io/en/latest/concepts/cli.html).

🆕  Added [Prophet](https://facebook.github.io/prophet/) support.

🆕  Need to upload additional files alongside your model? You can now use the `extras=` kwarg in `modelstore.upload()` to point modelstore to a file (or list of files) to upload as well.

## modelstore 0.0.71

🆕  Load models straight into memory! Model Store previously had `modelstore.download()` to download an artifact archive to a local path, it now also has `modelstore.load()` to load a model straight into memory.

🆕  Upload models from frameworks that are not (yet) supported by modelstore! The `modelstore.upload()` function now works if you give it a `model=` kwarg that is a path to a file.

🆕  Read a specific model's metadata with `modelstore.get_model_info()`

🆕  Added [Annoy](https://github.com/spotify/annoy), [ONNX](https://github.com/onnx/onnx), and [MXNet](https://mxnet.apache.org) (hybrid models) support.

## modelstore 0.0.65

🆕  Added model states, and updated listing models to listing by state.

🆕  Created a unified upload function. You can now use `modelstore.upload()` for all ML frameworks.

🆕  Added [Gensim](https://github.com/RaRe-Technologies/gensim) support.

🆕  Added Azure blob storage support.

🐛  Minor fixes to how modelstore uses env variables for the hosted storage, bug fixes for local file system storage.

🐛  Downgraded `requests` due to a version conflict with the version in Google Colab.

## modelstore 0.0.6

🆕  Added FastAI support.

🆕  Add support for scikit-learn pipelines.

## modelstore 0.0.52

🆕  Added PyTorch Lightning and LightGBM support.

🆕  Added a new type of storage: `ModelStore.from_api_key()`. If you're reading this and do not want to manage your own storage, get in touch with me for an API key.

🐛  Fixed bugs related to listing domains and the models inside of a domain.

🆕  Added skeleton functions for summary stats about training data; implemented feature importances for sklearn models.

## modelstore 0.0.4

🆕  Clean up how meta-data is generated

🆕  Add interactive authentication when using Google Colab

🐛  Upgraded dependencies to deal with an issue using `modelstore` in Colab

🆕  Added auto-extraction of model params and model info into the meta-data

## modelstore 0.0.3

🆕  Simplied the API to just requiring `upload()` (no more `create_archive()`).

## modelstore 0.0.2

🆕  Added models: `transformers`, `tensorflow`

🆕  Storage: downloading models via `download()`

🆕  Extended support to Python 3.6, 3.7, 3.8

🆕  Repo: added Github actions

## modelstore 0.0.1b

🆕  First release! Supports (and tested on) Python 3.7 only. ☢️

🆕  Storage: GCP buckets, AWS S3 buckets, file systems. Upload only!

🆕  Initial models: `catboost`, `keras`, `torch`, `sklearn`, `xgboost`

🆕  Meta-data: Python runtime, user, dependency versions, git hash
