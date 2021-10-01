# Adding support for a new machine learning framework

## Overview

Adding support for a new machine learning library entails:

1. Creating a class that extends `ModelManager`
2. Add the class to a dictionary of managers
3. Writing unit tests for the class you've added
4. Extending the `examples` to show people how to use `modelstore` with that framework

## Create a class that extends ModelManager

The `modelstore/models` directory contains one file per machine learning framework. Start by creating a new file for the framework you want to add - the convention is to name the file with the name of the machine learning framework (e.g. `modelstore/models/xgboost.py`).

Add a class in this file that extends `ModelManager` and implements all of `@abstractmethod` functions the class inherits. 

Note that we cannot guarantee that users of `modelstore` will have this machine learning framework installed in their environment. The code needs to be aware of this possibility, so whenever it needs to use framework-specific calls, it imports the library *inside* of that function. There are examples of this which have been explicitly called out with the `# pylint: disable=import-outside-toplevel` line.

Let's walk through what this looks like for `xgboost.py`. The class' constructor calls `super()` with its own name:

```python
class XGBoostManager(ModelManager):

    def __init__(self, storage: CloudStorage = None):
        super().__init__("xgboost", storage)
```

The `required_dependencies()` returns a list of Python libraries that are required by this machine learning framework. In most cases, the only dependency will be the library itself. The string values you add in here are the same as those you would put into a `requirements.txt` file.

```python
    @classmethod
    def required_dependencies(cls) -> list:
        return ["xgboost"]
```

The `optional_dependencies()` enumerates Python libraries that are not strictly required, but that it would be useful to collect information about if they are installed in the user's environment. You only need to implement this if there are dependencies that you want to add to the parent class' list.

```python
    @classmethod
    def optional_dependencies(cls) -> list:
        deps = super().optional_dependencies()
        return deps + ["sklearn"]
```

When users call `model_store.upload()` they will provide a set of key-value pairs that contain the model. The required key values are returned by `_required_kwargs()`:

```python
    def _required_kwargs(self):
        return ["model"]
```

The `matches_with()` function checks the types that have been passed into the `**kwargs` and returns whether they match with the current machine learning framework. In this case, the function returns whether the `model=` value is an `XGBModel`.

```python
    def matches_with(self, **kwargs) -> bool:
        # pylint: disable=import-outside-toplevel
        import xgboost as xgb

        return isinstance(kwargs.get("model"), xgb.XGBModel)
```

The `_get_functions()` function is called by model store in order to create a list of files that, together, are the model artefacts that need to be saved. Note: this function is returning a list of functions. Each one will be called with a `tmp_dir` argument and needs to return a string (a path to a single file) or a list (several paths to written files).

For `xgboost`, model store is saving the model, dumping the model, and storing the model config into separate files. The `_get_functions()` therefore returns three functions that need to be called. Each one has been prepopulated with some arguments using `functools.partial`.

```python
    def _get_functions(self, **kwargs) -> list:
        return [
            partial(save_model, model=kwargs["model"]),
            partial(dump_model, model=kwargs["model"]),
            partial(model_config, model=kwargs["model"]),
        ]
```

When model store is asked to store a model, it captures any meta data that it can about the model it is storing. This meta data is captured in the `_get_params()` function. This function is optional for now.

```python
    def _get_params(self, **kwargs) -> dict:
        """
        Returns a dictionary containing any model parameters
        that are available
        """
        return kwargs["model"].get_xgb_params()
```

Finally, the `load()` function does the reverse of the model artefact saving that was implemented in the functions that are returned by `_get_functions()`. Given a path to where model files exist on disk (`model_path`) and the `meta_data` dictionary that was saved when the model was uploaded, the `load()` function loads the model into memory and returns it.

```python
    def load(self, model_path: str, meta_data: dict) -> Any:
        # pylint: disable=import-outside-toplevel
        import xgboost as xgb

        model_types = {
            "XGBRegressor": xgb.XGBRegressor,
            "XGBClassifier": xgb.XGBClassifier,
            "XGBModel": xgb.XGBModel,
            # Future: other types
        }
        model_type = self._get_model_type(meta_data)
        if model_type not in model_types:
            raise ValueError(f"Cannot load xgboost model type: {model_type}")

        logger.debug("Loading xgboost model from %s", model_path)
        target = _model_file_path(model_path)
        model = model_types[model_type]()
        model.load_model(target)
        return model
```

## Add the class to a dictionary of managers

The `modelstore/models/managers.py` file contains a `ML_LIBRARIES` dictionary that keeps a mapping between each framework's name and the manager that we use to store/load models from it. 

This ensures that when a `modelstore` instance is created, it knows to check whether the machine learning framework you've added in the previous step exists in the user's environment.

Add an entry for your new framework, e.g.:

```python
# ...
from modelstore.models.xgboost import XGBoostManager

ML_LIBRARIES = {
    # ...
    "xgboost": XGBoostManager,
}
```

## Writing unit tests for the class you've added

The `tests/models/` directory contains all of the unit tests that we run for each machine learning framework.

The `modelstore` library typically includes tests that assert that:

* The `_model_info()`, `_model_data()`, and `_get_params()` functions returns the expected dictionary;
* The `_is_same_library()` returns True for the current machine learning framework, and False for others;
* The `_required_kwargs()` function returns the expected list;
* The `matches_with()` function returns True when the right key-value kwargs are passed, and False for incorrect pairs;
* The number of functions returned from `_get_functions()` is as expected; that individual functions save a model as they are expected to do;
* The `load()` function returns the same model that was previously saved

Feel free to include any additional unit tests that you think will be useful.

## Extend the `examples`

The examples directory shows `modelstore` users how to use the library.

In the `examples-by-ml-library/libraries` directory, you will find one Python file per machine learning framework. Create a new file there for the framework you're adding.

Implement two functions -- `train_and_upload(modelstore: ModelStore) -> dict`, which trains a model using the new framework and showcases how to upload it to the store, and `load_and_test(modelstore: ModelStore, model_id: str)` which showcases how to download/load a specific model with this framework from the store.

Add the new framework to `frameworks` in `run-all.sh`, and to the mapping of `EXAMPLES` in `main.py`.

You can then test it out with:

```bash
❯ python main.py --modelstore-in filesystem --ml-framework <the-new-framework>
```

