# modelstore examples

This directory contains examples of training models and storing them into a model store over different types of storage.

The Python script in `examples-by-ml-model` iterates over all of the supported ML frameworks and all of the supported storage types. For each pair, it trains a model, uploads it to storage, and then downloads/loads it back. 

The bash script `cli-examples` has exaples of how to run `python -m modelstore` commands.

## Pre-requisites

As with the main library, these scripts have been developed using [pyenv](https://github.com/pyenv/pyenv) and [pyenv-virtualenv](https://github.com/pyenv/pyenv-virtualenv).

## Set up - examples by ML model

Warning: the `examples-by-ml-model` virtual environment installs ALL of the machine learning frameworks that are supported by `modelstore`. In your own project, you will only need to install the machine learning frameworks that you need.

Start by `cd`'ing into the directory containing the example you want to run:

```bash
❯ cd examples-by-ml-model/
```

And then you can use this `Makefile` command that creates a new virtual environment
and installs all of the requirements:

```bash
❯ make pyenv
```

## Running all of the examples

After creating a virtual environment, you can run all of the examples using:

```bash
❯ make run
```

This will run all of the examples - you can expect it to take some time!

## Running a specific example

Start by `cd`'ing into the directory containing the example you want to run:

```bash
❯ cd examples-by-ml-model/
```

After creating a virtual environment, you can run all of the examples using:

```bash
❯ python main.py --modelstore-in $backend --ml-framework $framework
```
