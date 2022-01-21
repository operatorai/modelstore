# Contributing to modelstore

👋🏽 We'd love for you to contribute!

If you have any questions or ideas, feel free to reach out.

## Setup a virtual environment

This library has been developed using [pyenv](https://github.com/pyenv/pyenv)
and [pyenv-virtualenv](https://github.com/pyenv/pyenv-virtualenv), using the
requirements that are in `requirements.txt` and `requirements-dev.txt`.

This project has two requirements files:
* `requirements.txt` contains any dependencies that `modelstore` users must have in order to use `modelstore`. This should be as lightweight as possible. We do not require users to install every single machine learning library - just the ones that they want to use.
* `requirements-dev.txt` contains all of the dependencies that `modelstore` developers must have. This file does contain all of the machine learning frameworks that are supported by `modelstore` - they must be installed to enable running all of the unit tests. 

You can create a virtual environment using your favourite approach and
install all of those dependencies, or once you have set up `pyenv`, use this
 `Makefile` command that does the rest for you:

```bash
❯ make install
```

Will update `brew` and install `libomp` (required by `xgboost`).

It will then create a Python virtual environment, using `pyenv-virtualenv`,
and install all of the dependencies in the requirements files. If you want
to use a different version of Python, update the [bin/_config](bin/config) file.

Notes:
* I've seen trouble with installing `prophet`
* Even when `prophet` installs, there are sometimes issues with running `test_prophet`? See [this issue](https://github.com/facebook/prophet/issues/689). Uninstalling and reinstalling has worked for me.

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

For details, head over to the [README.md](examples/README.md) in the `examples` directory.

## Code formatting

This project uses `black` for code formatting.

## Add support for a new machine learning framework

The list of machine learning frameworks that are supported by `modelstore`
are found in the [modelstore/models/](modelstore/models) directory.

For a tutorial on how to add a new framework, refer to the [CONTRIBUTING.md](modelstore/models/CONTRIBUTING.md) guide in that directory.

## Add support for a new storage type

The list of storage types that are supported by `modelstore`
are found in the [modelstore/storage/](modelstore/storage) directory.

For a tutorial on how to add a new storage type, refer to the [CONTRIBUTING.md](modelstore/storage/CONTRIBUTING.md) guide in that directory.

## Adding other features

Do you have ideas for other features that it may be useful to add to `modelstore`?

Please get in touch by opening a [discussion](https://github.com/operatorai/modelstore/discussions) or [issue](https://github.com/operatorai/modelstore/issues) on Github.

