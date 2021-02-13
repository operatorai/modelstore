# Change log

## modelstore 0.0.52

Added PyTorch Lightning and LightGBM support.

Added a new type of storage: `ModelStore.from_api_key()`. If you're reading this and want an API key, get in touch!

Fixed bugs related to listing domains and the models inside of a domain.

Added skeleton functions for summary stats about training data; implemented feature importances for sklearn models. More to come on this front soon!

## modelstore 0.0.4

Clean up how meta-data is generated

Add interactive authentication when using Google Colab

Upgraded dependencies to deal with an issue using `modelstore` in Colab

Added auto-extraction of model params and model info into the meta-data

## modelstore 0.0.3

Simplied the API to just requiring `upload()` (no more `create_archive()`).

## modelstore 0.0.2

Added models: `transformers`, `tensorflow`

Storage: downloading models via `download()`

Extended support to Python 3.6, 3.7, 3.8

Repo: added Github actions

## modelstore 0.0.1b

First release! Supports (and tested on) Python 3.7 only. ☢️

Storage: GCP buckets, AWS S3 buckets, file systems. Upload only!

Initial models: `catboost`, `keras`, `torch`, `sklearn`, `xgboost`

Meta-data: Python runtime, user, dependency versions, git hash
