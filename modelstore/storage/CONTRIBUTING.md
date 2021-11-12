# Adding support for a new storage type

## Overview

Adding support for a new storage type entails:

1. Creating a class that extends `BlobStorage` or `CloudStorage`
2. Create a factory method in `modelstore.py`
3. Writing unit tests for the class you've added
4. Extending the `examples` to show people how to use `modelstore` with that storage

## Creating a class that extends BlobStorage or CloudStorage

The `modelstore/storage` directory contains one file per storage type. Start by creating a new file for the storage type you want to add - the convention is to name the file with the name of the storage (e.g. `modelstore/storage/gcloud.py`; note that, as of writing, `modelstore` is focusing on blob storage - like s3/cloud containers).

Add a class in this file that extends `BlobStorage` and implements all of `@abstractmethod` functions the class inherits. 

Note that we cannot guarantee that users of `modelstore` will have this storage type installed in their environment. The code needs to be aware of this possibility, so we check this at the top-level import.

## Create a factory method in `modelstore.py`

The main usage pattern for `modelstore` is to use a factory method:

```python
from modelstore import ModelStore

model_store = ModelStore.from_aws_s3("my-bucket")
```

To add support for a new storage layer, add a new factory method in `modelstore.ModelStore`. For example, here is the factory method for s3:

```python
    @classmethod
    def from_aws_s3(cls, bucket_name: Optional[str] = None, region: Optional[str] = None) -> "ModelStore":
        """Creates a ModelStore instance that stores models to an AWS s3
        bucket.

        This currently assumes that the s3 bucket already exists."""
        if not BOTO_EXISTS:
            raise ModuleNotFoundError("boto3 is not installed!")
        return ModelStore(
            storage=AWSStorage(bucket_name=bucket_name, region=region)
        )
```

Note that the arguments are all `Optional`. That is because `modelstore` enables retrieving these variables from the user's environment.

## Writing unit tests for the class you've added

The `tests/storage/` directory contains all of the unit tests that we run for each machine learning framework.

The `modelstore` library typically includes tests that assert that:

* The `validate()` function returns True or False, as expected;
* The `_push()` function stores a file to a given destination;
* The `_pull()` function downloads a file from a given destination
* The `_read_json_objects()` function returns a list of dictionaries;
* The `_read_json_object()` function returns a dictionary when the target file exists and `None` when it doesn't
* The `_storage_location()` function returns the expected meta data dictionary for that storage type
* The `_get_storage_location()` raises an error when the location doesn't exist

Feel free to include any additional unit tests that you think will be useful.

## Extend the `examples`

The examples directory shows `modelstore` users how to use the library.

In the `examples-by-storage/modelstores.py` file contains one function per storage type. Create a new function there for the storage type you're adding.

Extend the `create_model_store()` dictionary to add a mapping for the storage type you're adding; extend the list of storage options in `main.py`'s `click.option`s as well.

You can then test it out with:

```bash
❯ python main.py --modelstore-in <new-storage-type>
```

