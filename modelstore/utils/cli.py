import os
import sys
from enum import Enum
from typing import Optional

import click
from modelstore import ModelStore


class MessageStatus(Enum):

    """
    MessageStatus enumerates the different potential statuses
    that we can print messages about in the CLI, and their
    associated color
    """

    Sucess: str = "green"
    Failure: str = "red"
    Info: str = "blue"


def echo(message: str, status: MessageStatus):
    click.echo(click.style(message, fg=status.value))


def success(message: str):
    echo(message, MessageStatus.Sucess)


def failure(message: str):
    echo(message, MessageStatus.Failure)


def info(message: str):
    echo(message, MessageStatus.Info)


def assert_environ_exists(
    storage_type: str, keys: list, optional_keys: Optional[list]
):
    missing_required_keys = [k for k in keys if k not in os.environ]
    if len(missing_required_keys) != 0:
        missing_optional_keys = [
            k for k in optional_keys if k not in os.environ
        ]
        failure(
            f"❌ Failed to create {storage_type} modelstore.\nYour environment is missing:"
        )
        for k in missing_required_keys:
            failure(f"- {k} (required)")
        for k in missing_optional_keys:
            failure(f"- {k} (optional)")
        sys.exit(1)


def model_store_from_env() -> ModelStore:
    storage_type = os.environ["MODEL_STORE_STORAGE"]
    if storage_type == "aws":
        assert_environ_exists(
            "aws", ["MODEL_STORE_AWS_BUCKET"], ["MODEL_STORE_REGION"]
        )
        return ModelStore.from_aws_s3()
    if storage_type == "gcloud":
        assert_environ_exists(
            "aws", ["MODEL_STORE_GCP_PROJECT", "MODEL_STORE_GCP_BUCKET"], []
        )
        return ModelStore.from_gcloud()
    if storage_type == "azure":
        assert_environ_exists("aws", ["MODEL_STORE_AZURE_CONTAINER"], [])
        return ModelStore.from_azure()
    if storage_type == "filesystem":
        assert_environ_exists("aws", ["MODEL_STORE_ROOT"], [])
        return ModelStore.from_file_system()
    raise ValueError(f"Uknown modelstore type: {storage_type}")
