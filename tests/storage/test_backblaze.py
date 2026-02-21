#    Copyright 2020 Neal Lathia
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
from tempfile import TemporaryDirectory

import boto3
import pytest
from moto import mock_aws

from modelstore.metadata import metadata
from modelstore.storage.backblaze import BackblazeStorage
# pylint: disable=unused-import
from tests.storage.test_utils import (
    TEST_FILE_CONTENTS,
    TEST_FILE_NAME,
    file_contains_expected_contents,
    push_temp_file,
    push_temp_files,
    remote_file_path,
    remote_path
)

# pylint: disable=redefined-outer-name
# pylint: disable=protected-access
# pylint: disable=missing-function-docstring

_MOCK_BUCKET_NAME = "existing-bucket"


@pytest.fixture(autouse=True)
def mock_b2_env(monkeypatch):
    monkeypatch.setenv("B2_APPLICATION_KEY_ID", "testing")
    monkeypatch.setenv("B2_APPLICATION_KEY", "testing")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")


@pytest.fixture(autouse=True)
def moto_boto(mock_b2_env):
    with mock_aws():
        conn = boto3.resource("s3", region_name="us-east-1")
        conn.create_bucket(Bucket=_MOCK_BUCKET_NAME)
        yield conn


def get_file_contents(moto_boto, prefix):
    return (
        moto_boto.Object(_MOCK_BUCKET_NAME, prefix).get()["Body"].read().decode("utf-8")
    )


def _create_storage():
    client = boto3.client("s3", region_name="us-east-1")
    return BackblazeStorage(
        bucket_name=_MOCK_BUCKET_NAME,
        key_id="testing",
        application_key="testing",
        region="us-east-1",
        _client=client,
    )


def test_create_from_environment_variables(monkeypatch):
    monkeypatch.setenv("MODEL_STORE_B2_BUCKET", _MOCK_BUCKET_NAME)
    monkeypatch.setenv("B2_APPLICATION_KEY_ID", "testing")
    monkeypatch.setenv("B2_APPLICATION_KEY", "testing")
    try:
        _ = BackblazeStorage()
    except Exception:
        pytest.fail("Failed to initialise storage from env variables")


def test_create_fails_with_missing_environment_variables(monkeypatch):
    for key in BackblazeStorage.BUILD_FROM_ENVIRONMENT.get("required", []):
        monkeypatch.delenv(key, raising=False)
    with pytest.raises(KeyError):
        _ = BackblazeStorage()


@pytest.mark.parametrize(
    "bucket_name,validate_should_pass",
    [
        (
            "missing-bucket",
            False,
        ),
        (
            _MOCK_BUCKET_NAME,
            True,
        ),
    ],
)
def test_validate(bucket_name, validate_should_pass):
    client = boto3.client("s3", region_name="us-east-1")
    storage = BackblazeStorage(
        bucket_name=bucket_name,
        key_id="testing",
        application_key="testing",
        region="us-east-1",
        _client=client,
    )
    assert storage.validate() == validate_should_pass


def test_push(moto_boto):
    storage = _create_storage()
    result = push_temp_file(storage)

    assert result == remote_file_path()
    assert get_file_contents(moto_boto, result) == TEST_FILE_CONTENTS


def test_pull():
    storage = _create_storage()
    _ = push_temp_file(storage)

    with TemporaryDirectory() as tmp_dir:
        result = storage._pull(
            remote_file_path(),
            tmp_dir,
        )

        assert result == os.path.join(tmp_dir, TEST_FILE_NAME)
        assert os.path.exists(result)
        assert file_contains_expected_contents(result)


@pytest.mark.parametrize(
    "file_exists,should_call_delete",
    [
        (
            False,
            False,
        ),
        (
            True,
            True,
        ),
    ],
)
def test_remove(file_exists, should_call_delete):
    storage = _create_storage()
    if file_exists:
        _ = push_temp_file(storage)

    prefix = remote_file_path()
    assert storage._remove(prefix) == should_call_delete


def test_read_json_objects_ignores_non_json(tmp_path):
    storage = _create_storage()
    prefix = remote_path()
    push_temp_files(storage, prefix)

    items = storage._read_json_objects(prefix)
    assert len(items) == 1


def test_read_json_object_fails_gracefully(tmp_path):
    storage = _create_storage()
    remote = push_temp_file(storage, contents="not json")

    item = storage._read_json_object(remote)
    assert item is None


def test_storage_location():
    storage = _create_storage()
    prefix = remote_path()
    expected = metadata.Storage.from_bucket(
        storage_type="backblaze:b2",
        bucket=_MOCK_BUCKET_NAME,
        prefix=prefix,
    )
    assert storage._storage_location(prefix) == expected


@pytest.mark.parametrize(
    "meta_data,should_raise,result",
    [
        (
            metadata.Storage(
                type="",
                path=None,
                bucket=_MOCK_BUCKET_NAME,
                container=None,
                prefix="/path/to/file",
            ),
            False,
            "/path/to/file",
        ),
        (
            metadata.Storage(
                type="",
                path=None,
                bucket="a-different-bucket",
                container=None,
                prefix="/path/to/file",
            ),
            True,
            None,
        ),
    ],
)
def test_get_location(meta_data, should_raise, result):
    storage = _create_storage()
    if should_raise:
        with pytest.raises(ValueError):
            storage._get_storage_location(meta_data)
    else:
        assert storage._get_storage_location(meta_data) == result
