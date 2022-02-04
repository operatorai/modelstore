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
from typing import Optional

from modelstore.storage.blob_storage import BlobStorage
from modelstore.storage.util import environment
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

    NAME = "aws-s3"
    BUILD_FROM_ENVIRONMENT = {
        "required": [
            "MODEL_STORE_AWS_BUCKET",
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
        ],
        "optional": [
            "MODEL_STORE_REGION",
            "MODEL_STORE_ROOT_PREFIX",
        ],
    }

    def __init__(
        self,
        bucket_name: Optional[str] = None,
        region: Optional[str] = None,
        root_prefix: Optional[str] = None,
    ):
        super().__init__(["boto3"], root_prefix)
        # If arguments are None, try to populate them using environment variables
        self.bucket_name = environment.get_value(bucket_name, "MODEL_STORE_AWS_BUCKET")
        self.region = environment.get_value(
            region, "MODEL_STORE_REGION", allow_missing=True
        )
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

    def _pull(self, source: str, destination: str) -> str:
        logger.info("Downloading from: %s...", source)
        file_name = os.path.split(source)[1]
        destination = os.path.join(destination, file_name)
        self.client.download_file(self.bucket_name, source, destination)
        logger.debug("Finished: %s", destination)
        return destination

    def _remove(self, destination: str) -> bool:
        """ Removes a file from the destination path """
        try:
            self.client.head_object(Bucket=self.bucket_name, Key=destination)
            self.client.delete_object(Bucket=self.bucket_name, Key=destination)
            return True
        except ClientError as e:
            if int(e.response["Error"]["Code"]) != 404:
                raise
            logger.debug("Remote file does not exist: %s", destination)
            return False

    def _storage_location(self, prefix: str) -> dict:
        """ Returns a dict of the location the artifact was stored """
        return {
            "type": "aws:s3",
            "bucket": self.bucket_name,
            "prefix": prefix,
        }

    def _get_storage_location(self, meta: dict) -> str:
        """ Extracts the storage location from a meta data dictionary """
        if self.bucket_name != meta.get("bucket"):
            raise ValueError("Meta-data has a different bucket name")
        return meta["prefix"]

    def _read_json_objects(self, path: str) -> list:
        results = []
        objects = self.client.list_objects_v2(Bucket=self.bucket_name, Prefix=path)
        for version in objects.get("Contents", []):
            if not version["Key"].endswith(".json"):
                continue
            obj = self._read_json_object(version["Key"])
            if obj is not None:
                results.append(obj)
        return sorted_by_created(results)

    def _read_json_object(self, path: str) -> dict:
        obj = self.client.get_object(Bucket=self.bucket_name, Key=path)
        body = obj["Body"].read()
        try:
            return json.loads(body)
        except json.JSONDecodeError:
            return None
