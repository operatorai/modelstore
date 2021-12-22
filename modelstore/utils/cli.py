import os
import sys
from enum import Enum

import click
from modelstore import ModelStore
from modelstore.storage.aws import AWSStorage
from modelstore.storage.azure import AzureBlobStorage
from modelstore.storage.gcloud import GoogleCloudStorage
from modelstore.storage.local import FileSystemStorage

STORAGE_TYPES = {
    AWSStorage.NAME: AWSStorage,
    AzureBlobStorage.NAME: AzureBlobStorage,
    GoogleCloudStorage.NAME: GoogleCloudStorage,
    FileSystemStorage.NAME: FileSystemStorage,
}

MODEL_STORE_TYPES = {
    AWSStorage.NAME: ModelStore.from_aws_s3,
    AzureBlobStorage.NAME: ModelStore.from_azure,
    GoogleCloudStorage.NAME: ModelStore.from_gcloud,
    FileSystemStorage.NAME: ModelStore.from_file_system,
}


class MessageStatus(Enum):

    """
    MessageStatus enumerates the different potential statuses
    that we can print messages about in the CLI, and their
    associated color
    """

    Sucess: str = "green"
    Failure: str = "red"
    Info: str = "blue"


def _echo(message: str, status: MessageStatus):
    click.echo(click.style(message, fg=status.value), err=True)


def success(message: str):
    _echo(message, MessageStatus.Sucess)


def failure(message: str):
    _echo(message, MessageStatus.Failure)


def info(message: str):
    _echo(message, MessageStatus.Info)


def assert_environ_exists(storage_type: str, keys: dict):
    missing_required_keys = [
        k for k in keys.get("required", []) if k not in os.environ
    ]
    missing_optional_keys = [
        k for k in keys.get("optional", []) if k not in os.environ
    ]
    if len(missing_required_keys) != 0:
        failure(
            f"❌ Failed to create {storage_type} modelstore.\nYour environment is missing:"
        )
        for k in missing_required_keys:
            failure(f"- {k} (required)")
        for k in missing_optional_keys:
            failure(f"- {k} (optional)")
        sys.exit(1)


def model_store_from_env() -> ModelStore:
    if "MODEL_STORE_STORAGE" not in os.environ:
        failure("❌  No value for MODEL_STORE_STORAGE set in os.environ")
        sys.exit(1)

    storage_name = os.environ["MODEL_STORE_STORAGE"]
    if storage_name not in STORAGE_TYPES:
        failure(
            f"❌  Unknown storage name in MODEL_STORE_STORAGE: {storage_name}"
        )
        sys.exit(1)

    storage_type = STORAGE_TYPES[storage_name]
    assert_environ_exists(storage_name, storage_type.BUILD_FROM_ENVIRONMENT)
    info(f"Using model store with storage={storage_name}")
    return MODEL_STORE_TYPES[storage_name]()
