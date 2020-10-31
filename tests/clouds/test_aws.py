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
import json
import os
import time
from datetime import datetime

import boto3
import modelstore
from modelstore.clouds.aws import AWSStorage
from modelstore.clouds.util.paths import (get_archive_path, get_domain_path,
                                          get_metadata_path)
from moto import mock_s3

# pylint: disable=redefined-outer-name


def get_file_contents(conn, prefix):
    return (
        conn.Object("existing-bucket", prefix)
        .get()["Body"]
        .read()
        .decode("utf-8")
    )


@mock_s3
def test_validate():
    conn = boto3.resource("s3")
    conn.create_bucket(Bucket="existing-bucket")
    aws_model_store = AWSStorage(bucket_name="existing-bucket")
    assert aws_model_store.validate()
    aws_model_store = AWSStorage(bucket_name="missing-bucket")
    assert not aws_model_store.validate()


@mock_s3
def test_upload(tmp_path):
    source = os.path.join(tmp_path, "test-file.txt")
    with open(source, "w") as out:
        out.write("file-contents")

    conn = boto3.resource("s3")
    conn.create_bucket(Bucket="existing-bucket")
    aws_model_store = AWSStorage(bucket_name="existing-bucket")

    model_path = get_archive_path("test-domain", source)
    rsp = aws_model_store.upload("test-domain", source)
    assert rsp["type"] == "aws:s3"
    assert rsp["bucket"] == "existing-bucket"
    assert rsp["prefix"] == model_path
    assert get_file_contents(conn, model_path) == "file-contents"


@mock_s3
def test_set_meta_data():
    conn = boto3.resource("s3")
    conn.create_bucket(Bucket="existing-bucket")
    aws_model_store = AWSStorage(bucket_name="existing-bucket")

    meta_dict = {"key": "value"}
    meta_str = json.dumps(meta_dict)
    aws_model_store.set_meta_data("test-domain", "model-123", meta_dict)

    # Expected two upload
    meta_data = get_domain_path("test-domain")
    assert get_file_contents(conn, meta_data) == meta_str

    meta_data = get_metadata_path("test-domain", "model-123")
    assert get_file_contents(conn, meta_data) == meta_str


@mock_s3
def test_list_versions():
    conn = boto3.resource("s3")
    conn.create_bucket(Bucket="existing-bucket")
    aws_model_store = AWSStorage(bucket_name="existing-bucket")

    domain = "test-domain"
    for model in ["model-1", "model-2"]:
        created = datetime.now().strftime("%Y/%m/%d/%H:%M:%S")
        meta_data = {
            "model": {"domain": domain, "model_id": model,},
            "meta": {"created": created,},
            "modelstore": modelstore.__version__,
        }
        aws_model_store.set_meta_data(domain, model, meta_data)
        time.sleep(1)

    versions = aws_model_store.list_versions(domain)
    assert len(versions) == 2
    model_1_created = datetime.strptime(
        versions[0]["meta"]["created"], "%Y/%m/%d/%H:%M:%S"
    )
    model_2_created = datetime.strptime(
        versions[1]["meta"]["created"], "%Y/%m/%d/%H:%M:%S"
    )
    assert model_1_created > model_2_created


@mock_s3
def test_list_domains():
    conn = boto3.resource("s3")
    conn.create_bucket(Bucket="existing-bucket")
    aws_model_store = AWSStorage(bucket_name="existing-bucket")

    model = "test-model"
    for domain in ["domain-1", "domain-2"]:
        created = datetime.now().strftime("%Y/%m/%d/%H:%M:%S")
        meta_data = {
            "model": {"domain": domain, "model_id": model,},
            "meta": {"created": created,},
            "modelstore": modelstore.__version__,
        }
        aws_model_store.set_meta_data(domain, model, meta_data)
        time.sleep(1)

    versions = aws_model_store.list_domains()
    assert len(versions) == 2
    model_1_created = datetime.strptime(
        versions[0]["meta"]["created"], "%Y/%m/%d/%H:%M:%S"
    )
    model_2_created = datetime.strptime(
        versions[1]["meta"]["created"], "%Y/%m/%d/%H:%M:%S"
    )
    assert model_1_created > model_2_created
