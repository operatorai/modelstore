# Contributing to modelstore

👋🏽 We'd love for you to contribute!

If you have any questions or ideas, feel free to reach out. 

💬  Come and find us in the [MLOps Community Slack](https://go.mlops.community/slack)'s `#oss-modelstore` channel.

# Contribute your experience

If you've ever used `modelstore`, we've love for you to blog about your experience and link to your blog post in the repo's [README.md](https://github.com/operatorai/modelstore/blob/main/README.md) in the "Read more about modelstore" section.

You can also contribute to the [modelstore documentation](https://github.com/operatorai/modelstore-docs), which is in a separate Github repo.

# Contribute to the code base

To contribute to `modelstore`'s code base, we recommend taking the following journey:

1. Get familiar with the code base
2. Contribute fixes for [bugs or issues](https://github.com/operatorai/modelstore/issues)
3. Contribute new feature ideas from the [discussions](https://github.com/operatorai/modelstore/discussions) section 


## 👨🏽‍💻 Get familiar with the code base

### Setup a virtual environment

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
* I've seen trouble with installing `prophet` and have sometimes had to install it manually
* Even when `prophet` installs, there are sometimes issues with running `test_prophet`? See [this issue](https://github.com/facebook/prophet/issues/689). Uninstalling and reinstalling has worked for me.

### Run the tests

We're using `pytest` for the tests. You can type `pytest`, or use
this `Makefile` command:

```bash
❯ make test
```

### Run the examples

We have two types of examples:

* `examples-by-ml-library`, which has iterates over  machine learning model library and writes to three different model stores (GCP, AWS, file system)
* `cli-examples` shows how to use the `modelstore` command line interface

For details, head over to the [README.md](examples/README.md) in the `examples` directory.


## 🐛 Contribute fixes for bugs or issues

All of `modelstore`'s bugs are publicly tracked via [Github issues](https://github.com/operatorai/modelstore/issues). 


## 💡 Contribute new feature ideas

There are a variety of ideas that have been listed in the repo's [discussions](https://github.com/operatorai/modelstore/discussions) section. For example:

### Add support for a new machine learning framework

The list of machine learning frameworks that are supported by `modelstore`
are found in the [modelstore/models/](modelstore/models) directory.

For a tutorial on how to add a new framework, refer to the [CONTRIBUTING.md](modelstore/models/CONTRIBUTING.md) guide in that directory.

### Add support for a new storage type

The list of storage types that are supported by `modelstore`
are found in the [modelstore/storage/](modelstore/storage) directory.

For a tutorial on how to add a new storage type, refer to the [CONTRIBUTING.md](modelstore/storage/CONTRIBUTING.md) guide in that directory.

### Adding other features

Do you have ideas for other features that it may be useful to add to `modelstore`?

Please get in touch by opening a [discussion](https://github.com/operatorai/modelstore/discussions) or [issue](https://github.com/operatorai/modelstore/issues) on Github.
