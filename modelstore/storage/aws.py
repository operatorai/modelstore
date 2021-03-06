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

from modelstore.storage.blob_storage import BlobStorage
from modelstore.storage.util.paths import get_archive_path
from modelstore.storage.util.versions import sorted_by_created
from modelstore.utils.log import logger

try:
    import boto3
    from botocore.exceptions import ClientError

    BOTO_EXISTS = True
except ImportError:
    BOTO_EXISTS = False


class AWSStorage(BlobStorage):

    """
    AWS S3 Storage

    Assumes that you have `boto3` installed and configured
    https://boto3.amazonaws.com/v1/documentation/api/latest/guide/quickstart.html
    """

    def __init__(self, bucket_name: str, region: str = None):
        super().__init__(["boto3"])
        self.bucket_name = bucket_name
        self.region = region
        self.__client = None

    @property
    def client(self):
        try:
            if self.__client is None:
                self.__client = boto3.client("s3", region_name=self.region)
            return self.__client
        except ClientError:
            logger.error("Unable to create s3 client!")
            raise

    def validate(self) -> bool:
        logger.debug("Querying for buckets with prefix=%s...", self.bucket_name)
        try:
            resource = boto3.resource("s3")
            resource.meta.client.head_bucket(Bucket=self.bucket_name)
            return True
        except ClientError:
            logger.error("Unable to access bucket: %s", self.bucket_name)
            return False

    def _push(self, source: str, destination: str) -> str:
        logger.info("Uploading to: %s...", destination)
        self.client.upload_file(source, self.bucket_name, destination)
        logger.debug("Finished: %s", destination)
        return destination

    def _pull(self, source: dict, destination: str) -> str:
        prefix = _get_location(self.bucket_name, source)
        logger.info("Downloading from: %s...", prefix)
        file_name = os.path.split(prefix)[1]
        destination = os.path.join(destination, file_name)
        self.client.download_file(self.bucket_name, prefix, destination)
        logger.debug("Finished: %s", destination)
        return destination

    def upload(self, domain: str, model_id: str, local_path: str) -> dict:
        bucket_path = get_archive_path(domain, local_path)
        prefix = self._push(local_path, bucket_path)
        return _format_location(self.bucket_name, prefix)

    def _read_json_objects(self, path: str) -> list:
        results = []
        objects = self.client.list_objects_v2(
            Bucket=self.bucket_name, Prefix=path
        )
        for version in objects["Contents"]:
            if not version["Key"].endswith(".json"):
                # @TODO tighter controls here
                continue
            obj = self._read_json_object(version["Key"])
            results.append(obj)
        return sorted_by_created(results)

    def _read_json_object(self, path: str) -> dict:
        obj = self.client.get_object(Bucket=self.bucket_name, Key=path)
        body = obj["Body"].read()
        return json.loads(body)


def _format_location(bucket_name: str, prefix: str) -> dict:
    return {
        "type": "aws:s3",
        "bucket": bucket_name,
        "prefix": prefix,
    }


def _get_location(bucket_name, meta: dict) -> str:
    if bucket_name != meta["bucket"]:
        raise ValueError("Meta-data has a different bucket name")
    return meta["prefix"]
