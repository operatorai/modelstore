# Contributing to modelstore

👋🏽 We'd love for you to contribute!

If you have any questions or ideas, feel free to reach out.

## Setup

This package has been developed using [pyenv](https://github.com/pyenv/pyenv)
and [pyenv-virtualenv](https://github.com/pyenv/pyenv-virtualenv).

Once you have those installed, we have a `Makefile` command that does the
rest for you:

```bash
❯ make setup
```

Will update `brew` and install `libomp` (required by `xgboost`).

It will then create a Python 3.7.0 virtual environment, using `pyenv-virtualenv`,
and install all of the dependencies in the requirements files.

## Run the tests

We're using `pytest` for the tests. You can type `pytest`, or use
this `Makefile` command:

```bash
❯ make test
```

## Run the examples

We have two types of examples:

* `examples-by-ml-library`, which has 1 machine learning model library and writes to three different model stores (GCP, AWS, file system)
* `examples-by-storage` which has 2 machine learning libraries (`sklearn` and `xgboost`) and writes to one type of model store

To run a single example; for example, the `xgboost` one:

```bash
❯ cd examples/examples-by-ml-library/xgboost 
❯ make pyenv # or make pyenv-local if you have made local changes to the modelstore library
❯ make run # will run the Python script 3 times, with each model store type
```
