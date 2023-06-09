# Change log

## modelstore 0.0.79 ([June 2023](https://github.com/operatorai/modelstore/pull/243))

**🆕  New functionality**

Added support for any HDFS storage that you can access via [pydoop](https://crs4.github.io/pydoop/tutorial/hdfs_api.html#hdfs-api-tutorial): `ModelStore.from_hdfs()` [#241](https://github.com/operatorai/modelstore/pull/241), thanks [@sayandigital](https://github.com/sayandigital).

Updated the transformers manager: it no longer requires a `tokenizer` argument, so it can now be used to (for example) save/load SAM models [#238](https://github.com/operatorai/modelstore/pull/238) or DPT models [#239](https://github.com/operatorai/modelstore/pull/239). Thank you, [Cate in the MLOps Community](https://mlops-community.slack.com/archives/C0227QJCDS8/p1683293544101389)

**🐛  Bug fixes & general updates**

Fixed [issues with saving & loading GPT-2 models](https://github.com/operatorai/modelstore/issues/233) in [#234](https://github.com/operatorai/modelstore/pull/234), thank you [@sayandigital](https://github.com/sayandigital).

## modelstore 0.0.78 ([March 2023](https://github.com/operatorai/modelstore/pull/227))

**🆕  New functionality**

Added support for Python 3.9, and an example where a yolov5 model is loaded from the torch hub, saved to modelstore, and then loaded back from the model store [#225](https://github.com/operatorai/modelstore/pull/225), thanks to @h4mmer in the MLOps Community.

**🐛  Bug fixes & general updates**

You can now use `modelstore` with a local MinIO server that doesn't have SSL enabled [#223](workflows/investigate-minio-error), thanks to [@sspillard](https://github.com/sspillard).

`modelstore` will now raise an exception if the user is attempting to upload a file from their current working directory that is called `artifacts.tar.gz` -- because that's the file that `modelstore` wants to create ([#222](https://github.com/operatorai/modelstore/pull/222)).

## modelstore 0.0.77 ([December 2022](https://github.com/operatorai/modelstore/pull/216)) 

**🆕  New functionality**

Added support for any s3-compatible object storage that you can access via [MinIO](https://min.io/): `ModelStore.from_minio()`.

Added support for [Spark ML Lib](https://spark.apache.org/) models.

You can move model stores that you created in one file system directory to another one (e.g., making it a mounted volume in a container) and `modelstore` will continue to work ([#209](https://github.com/operatorai/modelstore/pull/209), thanks [@hauks96](https://github.com/operatorai/modelstore/issues/189)).

**🐛  Bug fixes & general updates**

Fixes an issue where model archives would be overwritten if models are uploaded too quickly ([#208](https://github.com/operatorai/modelstore/pull/208), thanks [@shsnyder](https://github.com/operatorai/modelstore/issues/197)).

## modelstore 0.0.76 ([October 2022](https://github.com/operatorai/modelstore/pull/205))

**🐛  Bug fixes & general updates**

A workaround for a security issue in the Python `tarfile` library was added ([#203](https://github.com/operatorai/modelstore/pull/203), thanks [@TrellixVulnTeam](https://github.com/TrellixVulnTeam)).

A security upgrade to the `protobuf` was merged ([#202](https://github.com/operatorai/modelstore/pull/202), thanks dependabot) - this impacts developers of this library only.

## modelstore 0.0.75 ([September 2022](https://github.com/operatorai/modelstore/pull/201))

**🆕  New functionality**

`modelstore` will issue a warning if you `load()` a model with a different version of Python than the version that was used to train the model ([#192](https://github.com/operatorai/modelstore/pull/192)).

You can now add any extra metadata to your model when uploading it, using `upload(domain, model, extra_metadata={ ... })` ([#185](https://github.com/operatorai/modelstore/pull/185)); if you want to upload extra _files_ with your model, then you should now use `extra_files=` instead of `extras=` ([#187](https://github.com/operatorai/modelstore/pull/187)).

**🐛  Bug fixes & general updates**

Fixed a bug when creating a GCP model store instance that reads from a public bucket ([#176](https://github.com/operatorai/modelstore/pull/176)).

Added support for storing xgboost Booster models ([#170](https://github.com/operatorai/modelstore/pull/170), [#177](https://github.com/operatorai/modelstore/pull/177)).

Refactored the library to use data classes for all model meta data ([#178](https://github.com/operatorai/modelstore/pull/178))

## modelstore 0.0.74 ([April 2022](https://github.com/operatorai/modelstore/pull/155))

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

The `list_versions()` function is deprecated and has been replaced with `list_models()` ([#132](https://github.com/operatorai/modelstore/pull/132))

Python 3.6 has passed its end-of-life, so this library is now tested with Python 3.7 and above.

## modelstore 0.0.73 ([February 2022](https://github.com/operatorai/modelstore/pull/121))

**🆕  New functionality**

You can upload multiple models to the same archive, if they don't share any keywords. For example `modelstore.upload(domain, model=sklearn_model, explainer=shap_explainer)` can be used to upload and download models and explainers together.

You can now set the root prefix of your model registry storage (thank you, [@cdknorow](https://github.com/cdknorow)!).

Added to the CLI functionality! You can now `python -m modelstore upload <domain> <model-file>` to upload a model. This requires you to [set environment variables](https://modelstore.readthedocs.io/en/latest/concepts/cli.html).

Added support for uploading [skorch](https://github.com/skorch-dev/skorch) models.

**🐛  Bug fixes**

Merged the model managers for `keras` and `tensorflow` into one.

## modelstore 0.0.72 ([November 2021](https://github.com/operatorai/modelstore/pull/100))

**🆕  New functionality**

Added support for uploading unsetting model states ([#82](https://github.com/operatorai/modelstore/issues/82) - hat tip to [@erosenthal-square](https://github.com/erosenthal-square) who opened an issue about this).

Added support for uploading [shap](https://shap.readthedocs.io/en/latest/generated/shap.Explainer.html) explainers.

Added CLI functionality! You can now `python -m modelstore download <domain> <model-id> <directory>` to download a model. This requires you to [set environment variables](https://modelstore.readthedocs.io/en/latest/concepts/cli.html).

Added [Prophet](https://facebook.github.io/prophet/) support.

Need to upload additional files alongside your model? You can now use the `extras=` kwarg in `modelstore.upload()` to point modelstore to a file (or list of files) to upload as well.

**🐛  Bug fixes**

Saving complex sklearn pipelines was [raising a TypeError](https://github.com/operatorai/modelstore/issues/96). This is because the `get_params()` function, which `modelstore` uses to save meta data about the model, returns a lot of things that are not JSON serializable. For now, I've patched this by not returning metadata for `sklearn.pipeline.Pipeline` models.

Colab is currently running `fastai==1.0.61`, while `modelstore` was designed for `fastai>2`, so things would [break in Colab Notebooks](https://github.com/operatorai/modelstore/issues/95) due to the different import paths in the two versions of `fastai`: the import paths are now version-dependent..

Updated the library so that PyTorch models can be uploaded [without an optimizer](https://github.com/operatorai/modelstore/issues/94). This is useful for uploading pretrained embedding models!

Fixed [a logging bug when trying to download the latest model](https://github.com/operatorai/modelstore/issues/91) in a domain - hat tip to [@erosenthal-square](https://github.com/erosenthal-square) who found the issue.

Fixed an `ImportError` bug when trying to use `modelstore` on an instance [that does not have git installed](https://github.com/operatorai/modelstore/pull/86).

## modelstore 0.0.71 ([September 2021](https://github.com/operatorai/modelstore/pull/78))

**🆕  New functionality**

Load models straight into memory! Model Store previously had `modelstore.download()` to download an artifact archive to a local path, it now also has `modelstore.load()` to load a model straight into memory.

Upload models from frameworks that are not (yet) supported by modelstore! The `modelstore.upload()` function now works if you give it a `model=` kwarg that is a path to a file.

Read a specific model's metadata with `modelstore.get_model_info()`

Added [Annoy](https://github.com/spotify/annoy), [ONNX](https://github.com/onnx/onnx), and [MXNet](https://mxnet.apache.org) (hybrid models) support.

## modelstore 0.0.65 (July 2021)

**🆕  New functionality**

Added model states, and updated listing models to listing by state.

Created a unified upload function. You can now use `modelstore.upload()` for all ML frameworks.

Added [Gensim](https://github.com/RaRe-Technologies/gensim) support.

Added Azure blob storage support.

**🐛  Bug fixes**

Minor fixes to how modelstore uses env variables for the hosted storage, bug fixes for local file system storage.

Downgraded `requests` due to a version conflict with the version in Google Colab.

## modelstore 0.0.6 ([March 2021](https://github.com/operatorai/modelstore/pull/29))

**🆕  New functionality**

Added FastAI support.

Add support for scikit-learn pipelines.

## modelstore 0.0.52 ([February 2021](https://github.com/operatorai/modelstore/pull/24))

**🆕  New functionality**

Added PyTorch Lightning and LightGBM support.

[Update - this has been removed] Added a new type of storage: `ModelStore.from_api_key()`. If you're reading this and do not want to manage your own storage, get in touch with me for an API key.

Added skeleton functions for summary stats about training data; implemented feature importances for sklearn models.

**🐛  Bug fixes**

Fixed bugs related to listing domains and the models inside of a domain.

## modelstore 0.0.4 ([December 2020](https://github.com/operatorai/modelstore/pull/17))

**🆕  New functionality**

Clean up how meta-data is generated

Add interactive authentication when using Google Colab

Added auto-extraction of model params and model info into the meta-data

**🐛  Bug fixes**

Upgraded dependencies to deal with an issue using `modelstore` in Colab

## modelstore 0.0.3 ([November 2020](https://github.com/operatorai/modelstore/pull/9))

**🆕  New functionality**

Simplied the API to just requiring `upload()` (no more `create_archive()`).

## modelstore 0.0.2 ([September 2020](https://github.com/operatorai/modelstore/pull/7))

**🆕  New functionality**

Added models: `transformers`, `tensorflow`

Storage: downloading models via `download()`

Extended support to Python 3.6, 3.7, 3.8

Repo: added Github actions

## modelstore 0.0.1b (September 2020)

**🆕  First release!**

Supports (and tested on) Python 3.7 only. ☢️

Storage: GCP buckets, AWS S3 buckets, file systems. Upload only!

Initial models: `catboost`, `keras`, `torch`, `sklearn`, `xgboost`

Meta-data: Python runtime, user, dependency versions, git hash
