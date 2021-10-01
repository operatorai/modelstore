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

It will then create a Python virtual environment, using `pyenv-virtualenv`,
and install all of the dependencies in the requirements files. If you want
to use a different version of Python, update the [bin/_config](bin/config) file.

## Run the tests

We're using `pytest` for the tests. You can type `pytest`, or use
this `Makefile` command:

```bash
❯ make test
```

## Run the examples

We have two types of examples:

* `examples-by-ml-library`, which has iterates over  machine learning model library and writes to three different model stores (GCP, AWS, file system)
* `examples-by-storage` which has 2 machine learning libraries (`sklearn` and `xgboost`) and writes to one type of model store
