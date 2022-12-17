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

from modelstore.metadata import metadata
from modelstore.storage.blob_storage import BlobStorage
from modelstore.storage.util import environment
from modelstore.storage.util.versions import sorted_by_created
from modelstore.utils.log import logger
from modelstore.utils.exceptions import FilePullFailedException

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
            "MODEL_STORE_AWS_ROOT_PREFIX",
        ],
    }

    def __init__(
        self,
        bucket_name: Optional[str] = None,
        region: Optional[str] = None,
        root_prefix: Optional[str] = None,
    ):
        super().__init__(["boto3"], root_prefix, "MODEL_STORE_AWS_ROOT_PREFIX")
        # If arguments are None, try to populate them using environment variables
        self.bucket_name = environment.get_value(bucket_name, "MODEL_STORE_AWS_BUCKET")
        self.region = environment.get_value(
            region, "MODEL_STORE_REGION", allow_missing=True
        )
        self.__client = None

    @property
    def client(self):
        """Returns the boto s3 client"""
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

    def _push(self, file_path: str, prefix: str) -> str:
        logger.info("Uploading to: %s...", prefix)
        self.client.upload_file(file_path, self.bucket_name, prefix)
        return prefix

    def _pull(self, prefix: str, dir_path: str) -> str:
        try:
            logger.debug("Downloading from: %s...", prefix)
            file_name = os.path.split(prefix)[1]

            destination = os.path.join(dir_path, file_name)
            self.client.download_file(self.bucket_name, prefix, destination)
            return destination
        except ClientError as exc:
            if int(exc.response["Error"]["Code"]) == 404:
                raise FilePullFailedException(exc) from exc
            raise exc

    def _remove(self, prefix: str) -> bool:
        """Removes a file from the destination path"""
        try:
            self.client.head_object(Bucket=self.bucket_name, Key=prefix)
            logger.debug("Deleting: %s...", prefix)
            self.client.delete_object(Bucket=self.bucket_name, Key=prefix)
            return True
        except ClientError as exc:
            if int(exc.response["Error"]["Code"]) == 404:
                logger.debug("Remote file does not exist: %s", prefix)
                return False
            raise

    def _storage_location(self, prefix: str) -> metadata.Storage:
        """Returns a dict of the location the artifact was stored"""
        return metadata.Storage.from_bucket(
            storage_type="aws:s3",
            bucket=self.bucket_name,
            prefix=prefix,
        )

    def _get_storage_location(self, meta_data: metadata.Storage) -> str:
        """Extracts the storage location from a meta data dictionary"""
        if self.bucket_name != meta_data.bucket:
            # @TODO: downgrade to a warning if the file exists
            raise ValueError("Meta-data has a different bucket name")
        return meta_data.prefix

    def _read_json_objects(self, prefix: str) -> list:
        logger.debug("Listing files in: %s/%s", self.bucket_name, prefix)
        results = []
        objects = self.client.list_objects_v2(Bucket=self.bucket_name, Prefix=prefix)
        for version in objects.get("Contents", []):
            object_path = version["Key"]
            if not object_path.endswith(".json"):
                logger.debug("Skipping non-json file: %s", object_path)
                continue
            if os.path.split(object_path)[0] != prefix:
                # We don't want to read files in a sub-prefix
                logger.debug("Skipping file in sub-prefix: %s", object_path)
                continue

            obj = self._read_json_object(object_path)
            if obj is not None:
                results.append(obj)
        return sorted_by_created(results)

    def _read_json_object(self, prefix: str) -> dict:
        logger.debug("Reading: %s/%s", self.bucket_name, prefix)
        obj = self.client.get_object(Bucket=self.bucket_name, Key=prefix)
        body = obj["Body"].read()
        try:
            return json.loads(body)
        except json.JSONDecodeError:
            return None
