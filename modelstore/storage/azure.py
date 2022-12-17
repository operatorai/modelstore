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
    from azure.storage.blob import BlobServiceClient
    from azure.core.exceptions import ResourceNotFoundError

    AZURE_EXISTS = True
except ImportError:
    AZURE_EXISTS = False


class AzureBlobStorage(BlobStorage):

    """
    Azure Blob Storage

    Assumes that azure.storage.blob is installed and configured
    and that the Azure Container already exists
    """

    NAME = "azure-container"
    BUILD_FROM_ENVIRONMENT = {
        "required": [
            "MODEL_STORE_AZURE_CONTAINER",
            "AZURE_ACCOUNT_NAME",
            "AZURE_ACCESS_KEY",
            "AZURE_STORAGE_CONNECTION_STRING",
        ],
        "optional": [
            "MODEL_STORE_AZURE_ROOT_PREFIX",
        ],
    }

    def __init__(
        self,
        container_name: Optional[str] = None,
        root_prefix: Optional[str] = None,
        client: "BlobServiceClient" = None,
        environ_key: str = "AZURE_STORAGE_CONNECTION_STRING",
    ):
        super().__init__(
            ["azure.storage.blob"], root_prefix, "MODEL_STORE_AZURE_ROOT_PREFIX"
        )
        # If arguments are None, try to populate them using environment variables
        self.container_name = environment.get_value(
            container_name, "MODEL_STORE_AZURE_CONTAINER"
        )
        self.connection_string_key = environ_key
        self.__client = client

    @property
    def client(self) -> "BlobServiceClient":
        """Returns the azure client"""
        if not AZURE_EXISTS:
            raise ImportError("Please install azure-storage-blob")
        if self.connection_string_key not in os.environ:
            raise Exception(f"{self.connection_string_key} is not in os.environ")
        if self.__client is None:
            connect_str = os.environ[self.connection_string_key]
            self.__client = BlobServiceClient.from_connection_string(connect_str)
        return self.__client

    def _container_client(self):
        return self.client.get_container_client(self.container_name)

    def _blob_client(self, blob_name: str):
        blob_client = self._container_client().get_blob_client(blob_name)
        chunk_size = 2097152  # 1024 * 1024 B * 2 = 2 MB

        # The maximum chunk size for uploading a block blob in chunks
        blob_client.max_block_size = chunk_size

        # If the blob size is larger than max_single_put_size, the blob will be uploaded in chunks
        blob_client.max_single_put_size = chunk_size

        # The maximum size for a blob to be downloaded in a single call
        blob_client.max_single_get_size = chunk_size
        blob_client.max_chunk_get_size = chunk_size
        return blob_client

    def validate(self) -> bool:
        """Checks that the Azure container exists"""
        logger.debug("Querying for containers with name=%s...", self.container_name)
        return self._container_client().exists()

    def _push(self, file_path: str, prefix: str) -> str:
        logger.info("Uploading to: %s...", prefix)
        blob_client = self._blob_client(prefix)

        with open(file_path, "rb") as data:
            blob_client.upload_blob(data, overwrite=True)
        return prefix

    def _pull(self, prefix: str, dir_path: str) -> str:
        """Pulls a model to a destination"""
        try:
            logger.debug("Downloading from: %s...", prefix)
            blob_client = self._blob_client(prefix)
            target = os.path.join(dir_path, os.path.split(prefix)[1])
            with open(target, "wb") as download_file:
                download_file.write(blob_client.download_blob().readall())
            return target
        except ResourceNotFoundError as exc:
            raise FilePullFailedException(exc) from exc

    def _remove(self, prefix: str) -> bool:
        """Removes a file from the destination path"""
        blob_client = self._blob_client(prefix)
        if not blob_client.exists():
            logger.debug("Remote file does not exist: %s", prefix)
            return False
        blob_client.delete_blob()
        return True

    def _storage_location(self, prefix: str) -> metadata.Storage:
        """Returns a dict of the location the artifact was stored"""
        return metadata.Storage.from_container(
            storage_type="azure:blob-storage",
            container=self.container_name,
            prefix=prefix,
        )

    def _get_storage_location(self, meta_data: metadata.Storage) -> str:
        """Extracts the storage location from a meta data dictionary"""
        if self.container_name != meta_data.container:
            raise ValueError("Meta-data has a different container name")
        return meta_data.prefix

    def _read_json_objects(self, prefix: str) -> list:
        logger.debug("Listing files in: %s/%s", self.container_name, prefix)
        results = []
        blobs = self._container_client().list_blobs(name_starts_with=prefix + "/")
        for blob in blobs:
            if not blob.name.endswith(".json"):
                logger.debug("Skipping non-json file: %s", blob.name)
                continue
            if os.path.split(blob.name)[0] != prefix:
                # We don't want to read files in a sub-prefix
                logger.debug("Skipping file in sub-prefix: %s", blob.name)
                continue
            blob_client = self._blob_client(blob)
            obj = blob_client.download_blob().readall()
            if obj is not None:
                results.append(json.loads(obj))
        return sorted_by_created(results)

    def _read_json_object(self, prefix: str) -> dict:
        """Returns a dictionary of the JSON stored in a given path"""
        blob_client = self._blob_client(prefix)
        body = blob_client.download_blob().readall()
        try:
            return json.loads(body)
        except json.JSONDecodeError:
            return None
