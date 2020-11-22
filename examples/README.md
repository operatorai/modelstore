# Operator AI Examples

This directory contains examples of training models and storing them into a model store.

## Running the examples

As with the main library, these scripts have been developed using [pyenv](https://github.com/pyenv/pyenv) and [pyenv-virtualenv](https://github.com/pyenv/pyenv-virtualenv).

Once you have those installed, `cd` into the directory containing the example you want to run:

```bash
❯ cd examples-by-ml-model/sklearn
```

And then you can use this `Makefile` command that creates a new virtual environment
and installs all of the requirements:

```bash
❯ make pyenv
```

Finally, to run the example itself, run `main.py` with the type of backend storage you want
the model store to use:

```bath
❯ python main.py --storage [filestore|gcloud|aws]
```
