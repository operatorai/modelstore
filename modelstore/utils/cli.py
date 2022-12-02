#    Copyright 2022 Neal Lathia
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
import os
import sys
from enum import Enum
import click

from modelstore import ModelStore
from modelstore.storage.aws import AWSStorage
from modelstore.storage.azure import AzureBlobStorage
from modelstore.storage.gcloud import GoogleCloudStorage
from modelstore.storage.local import FileSystemStorage
from modelstore.storage.minio import MinIOStorage


STORAGE_TYPES = {
    AWSStorage.NAME: AWSStorage,
    AzureBlobStorage.NAME: AzureBlobStorage,
    GoogleCloudStorage.NAME: GoogleCloudStorage,
    FileSystemStorage.NAME: FileSystemStorage,
    MinIOStorage.NAME: MinIOStorage,
}

MODEL_STORE_TYPES = {
    AWSStorage.NAME: ModelStore.from_aws_s3,
    AzureBlobStorage.NAME: ModelStore.from_azure,
    GoogleCloudStorage.NAME: ModelStore.from_gcloud,
    FileSystemStorage.NAME: ModelStore.from_file_system,
    MinIOStorage.NAME: ModelStore.from_minio,
}


class MessageStatus(Enum):

    """
    MessageStatus enumerates the different potential statuses
    that we can print messages about in the CLI, and their
    associated color
    """

    SUCCESS: str = "green"
    FAILURE: str = "red"
    INFO: str = "blue"


def _echo(message: str, status: MessageStatus):
    click.echo(click.style(message, fg=status.value), err=True)


def success(message: str):
    """Echos a message in green"""
    _echo(message, MessageStatus.SUCCESS)


def failure(message: str):
    """Echos a message in red"""
    _echo(message, MessageStatus.FAILURE)


def info(message: str):
    """Echos a message in blue"""
    _echo(message, MessageStatus.INFO)


def assert_environ_exists(storage_type: str, keys: dict):
    """Checks that environment variables that are required to use the modelstore CLI
    are set. If not, logs the failure and exits."""
    missing_required_keys = [k for k in keys.get("required", []) if k not in os.environ]
    missing_optional_keys = [k for k in keys.get("optional", []) if k not in os.environ]
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
    """Builds a modelstore instance from environment variables."""
    if "MODEL_STORE_STORAGE" not in os.environ:
        failure("❌  No value for MODEL_STORE_STORAGE set in os.environ")
        sys.exit(1)

    storage_name = os.environ["MODEL_STORE_STORAGE"]
    if storage_name not in STORAGE_TYPES:
        failure(f"❌  Unknown storage name in MODEL_STORE_STORAGE: {storage_name}")
        sys.exit(1)

    storage_type = STORAGE_TYPES[storage_name]
    assert_environ_exists(storage_name, storage_type.BUILD_FROM_ENVIRONMENT)
    info(f"Using model store with storage={storage_name}")
    return MODEL_STORE_TYPES[storage_name]()
