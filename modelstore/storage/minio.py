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
    from minio import Minio
    from minio.error import InvalidResponseError, S3Error

    MINIO_EXISTS = True
except ImportError:
    MINIO_EXISTS = False


class MinIOStorage(BlobStorage):

    """
    MinIO Storage

    Assumes that you have `minio` installed
    https://github.com/minio/minio-py
    https://min.io/docs/minio/kubernetes/upstream/index.html
    """

    NAME = "minio"
    BUILD_FROM_ENVIRONMENT = {
        "required": [
            "MODEL_STORE_MINIO_BUCKET",
            "MINIO_ACCESS_KEY",
            "MINIO_SECRET_KEY",
        ],
        "optional": [
            "MODEL_STORE_MINIO_ROOT_PREFIX",
            "MINIO_ENDPOINT",
        ],
    }

    def __init__(
        self,
        endpoint: Optional[str] = None,
        access_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        bucket_name: Optional[str] = None,
        root_prefix: Optional[str] = None,
        secure: Optional[bool] = True,
        client: "Minio" = None,
    ):
        super().__init__(["minio"], root_prefix, "MODEL_STORE_MINIO_ROOT_PREFIX")
        # If arguments are None, try to populate them using environment variables
        self.bucket_name = environment.get_value(
            bucket_name, "MODEL_STORE_MINIO_BUCKET"
        )
        self.endpoint = environment.get_value(
            endpoint, "MINIO_ENDPOINT", allow_missing=True
        )
        if self.endpoint is None:
            self.endpoint = "s3.amazonaws.com"
        self.__client = client
        if client is not None:
            # access / secret not required when the client is passed in
            return
        self.access_key = environment.get_value(access_key, "MINIO_ACCESS_KEY")
        self.secret_key = environment.get_value(secret_key, "MINIO_SECRET_KEY")
        self.secure = secure

    @property
    def client(self):
        """Returns the minio client"""
        if self.__client is None:
            self.__client = Minio(
                self.endpoint,
                access_key=self.access_key,
                secret_key=self.secret_key,
                secure=self.secure,
            )
        return self.__client

    def validate(self) -> bool:
        logger.debug("Querying for bucket=%s...", self.bucket_name)
        return self.client.bucket_exists(self.bucket_name)

    def _push(self, file_path: str, prefix: str) -> str:
        logger.info("Uploading to: %s...", prefix)
        with open(file_path, "rb") as file_data:
            file_stat = os.stat(file_path)
            self.client.put_object(
                self.bucket_name,
                prefix,
                file_data,
                file_stat.st_size,
            )
        return prefix

    def _pull(self, prefix: str, dir_path: str) -> str:
        try:
            logger.debug("Downloading from: %s...", prefix)
            file_name = os.path.split(prefix)[1]
            destination = os.path.join(dir_path, file_name)

            self.client.fget_object(self.bucket_name, prefix, destination)
            return destination
        except (InvalidResponseError, S3Error) as exc:
            raise FilePullFailedException(exc) from exc

    def _remove(self, prefix: str) -> bool:
        """Removes a file from the destination path"""
        objects = [
            obj
            for obj in self.client.list_objects(
                self.bucket_name,
                prefix,
                recursive=False,
            )
        ]
        if len(objects) == 0:
            logger.debug("Remote file does not exist: %s", prefix)
            return False
        logger.debug("Deleting: %s...", prefix)
        self.client.remove_object(self.bucket_name, prefix)
        return True

    def _storage_location(self, prefix: str) -> metadata.Storage:
        """Returns a dict of the location the artifact was stored"""
        return metadata.Storage.from_bucket(
            storage_type=f"minio:{self.endpoint}",
            bucket=self.bucket_name,
            prefix=prefix,
        )

    def _get_storage_location(self, meta_data: metadata.Storage) -> str:
        """Extracts the storage location from a meta data dictionary"""
        if not meta_data.type.endswith(self.endpoint):
            raise ValueError("Meta-data has a different endpoint name")
        if self.bucket_name != meta_data.bucket:
            # @TODO: downgrade to a warning if the file exists
            raise ValueError("Meta-data has a different bucket name")
        return meta_data.prefix

    def _read_json_objects(self, prefix: str) -> list:
        logger.debug("Listing files in: %s/%s", self.bucket_name, prefix)
        results = []
        objects = self.client.list_objects(
            self.bucket_name,
            prefix,
            recursive=True,
        )
        for obj in objects:
            if not obj.object_name.endswith(".json"):
                logger.debug("Skipping non-json file: %s", obj.object_name)
                continue
            if os.path.split(obj.object_name)[0] != prefix:
                # We don't want to read files in a sub-prefix
                logger.debug("Skipping file in sub-prefix: %s", obj.object_name)
                continue
            obj = self._read_json_object(obj.object_name)
            if obj is not None:
                results.append(obj)
        return sorted_by_created(results)

    def _read_json_object(self, prefix: str) -> dict:
        logger.debug("Reading: %s/%s", self.bucket_name, prefix)
        obj = self.client.get_object(self.bucket_name, prefix)
        lines = obj.readlines()
        if len(lines) == 0:
            return None
        try:
            return json.loads(lines[0])
        except json.JSONDecodeError:
            return None
