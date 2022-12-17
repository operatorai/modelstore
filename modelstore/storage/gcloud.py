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
import warnings

from modelstore.metadata import metadata
from modelstore.storage.blob_storage import BlobStorage
from modelstore.storage.util import environment
from modelstore.storage.util.versions import sorted_by_created
from modelstore.utils.log import logger
from modelstore.utils.exceptions import FilePullFailedException

try:
    from google.auth.exceptions import DefaultCredentialsError
    from google.api_core.exceptions import NotFound, Forbidden
    from google.cloud import storage

    # pylint: disable=protected-access
    storage.blob._DEFAULT_CHUNKSIZE = 2097152  # 1024 * 1024 B * 2 = 2 MB
    storage.blob._MAX_MULTIPART_SIZE = 2097152  # 2 MB

    GCLOUD_EXISTS = True
except ImportError:
    GCLOUD_EXISTS = False


class GoogleCloudStorage(BlobStorage):

    """
    Google Cloud Storage

    Assumes that google.cloud.storage is installed and configured
    """

    NAME = "google-cloud-storage"
    BUILD_FROM_ENVIRONMENT = {
        "required": ["MODEL_STORE_GCP_PROJECT", "MODEL_STORE_GCP_BUCKET"],
        "optional": ["MODEL_STORE_GCP_ROOT_PREFIX"],
    }

    def __init__(
        self,
        project_name: Optional[str] = None,
        bucket_name: Optional[str] = None,
        root_prefix: Optional[str] = None,
        client: "storage.Client" = None,
        is_anon_client: bool = False,
    ):
        super().__init__(
            ["google.cloud.storage"],
            root_prefix,
            "MODEL_STORE_GCP_ROOT_PREFIX",
        )
        # If arguments are None, try to populate them using environment variables
        self.bucket_name = environment.get_value(bucket_name, "MODEL_STORE_GCP_BUCKET")

        # If the project_name is not given and not available as an environment
        # variable, this storage client will connect to the bucket_name anonymously
        # and will be read-only
        self.project_name = environment.get_value(
            project_name, "MODEL_STORE_GCP_PROJECT", allow_missing=True
        )

        # This can be set in the constructor to faciliate unit testing
        self.is_anon_client = is_anon_client
        self.__client = client

    @property
    def client(self) -> "storage.Client":
        """Returns a gcloud storage client"""
        if not GCLOUD_EXISTS:
            # Google cloud is not installed or cannot be imported
            raise ImportError("Please install google-cloud-storage")

        if self.__client is not None:
            # Return the existing client, which may have been passed
            # as an argument to GoogleCloudStorage's init()
            return self.__client

        if self.project_name is not None:
            try:
                # If the user gives a project name, return an
                # explicit storage.Client
                self.is_anon_client = False
                self.__client = storage.Client(self.project_name)
            except DefaultCredentialsError:
                try:
                    # Try to authenticate, if in a Colab notebook
                    # pylint: disable=no-name-in-module,import-error,import-outside-toplevel
                    from google.colab import auth

                    auth.authenticate_user()
                    self.is_anon_client = False
                    self.__client = storage.Client(self.project_name)
                except ModuleNotFoundError:
                    logger.warning(
                        "Missing credentials: https://cloud.google.com/docs/authentication/getting-started#command-line"  # noqa
                    )
                    warnings.warn(
                        "No credentials given, falling back to anonymous access."
                    )
        else:
            # If no project name is given, create a read- and list-only client
            # Note: uploads will not work
            self.is_anon_client = True
            self.__client = storage.Client.create_anonymous_client()
        return self.__client

    @property
    def bucket(self):
        """The bucket where model artifacts will be stored"""
        if self.is_anon_client:
            return self.client.bucket(bucket_name=self.bucket_name)
        try:
            # Try to retrieve a bucket (this makes an API request)
            return self.client.get_bucket(self.bucket_name)
        except (NotFound, Forbidden):
            # NotFound/Forbidden can be raised when both
            # (a) The self.project_name is not None; e.g. it was sourced
            # from environment variables in the constructor
            # (b) The bucket is a public, read-only bucket
            return self.client.bucket(bucket_name=self.bucket_name)

    def validate(self) -> bool:
        """Validates that the cloud bucket exists"""
        try:
            logger.debug("Checking if bucket exists (%s)...", self.bucket_name)
            if not self.bucket.exists():
                logger.error(
                    "Bucket '%s' does not exist or is not accessible to your client.",
                    self.bucket_name,
                )
                return False
            return True
        except (ValueError, Forbidden):
            # ValueError: the anonymous client appears to fail when using
            # bucket.exists() for buckets that _do_ exist.
            # https://github.com/operatorai/modelstore/issues/173
            # Forbidden: the non-anonymous client appears to fail when using
            # bucket.exists() on public/read-only buckets.
            # In both cases, we fall back on listing the contents of the bucket
            # to check whether we can read its contents
            try:
                logger.debug("Querying for blobs in bucket (%s)...", self.bucket_name)
                return (
                    list(self.client.list_blobs(self.bucket_name, max_results=1))
                    is not None
                )
            except NotFound:
                logger.error(
                    "Cannot list blobs in '%s' with an anonymous client.",
                    self.bucket_name,
                )
                return False

    def _push(self, file_path: str, prefix: str) -> str:
        if self.is_anon_client:
            raise NotImplementedError(
                "File upload is only supported for authenticated clients."
            )
        logger.info("Uploading to: %s...", prefix)
        blob = self.bucket.blob(prefix)

        ## For slow upload speed
        # https://stackoverflow.com/questions/61001454/why-does-upload-from-file-google-cloud-storage-function-throws-timeout-error

        with open(file_path, "rb") as f:
            blob.upload_from_file(f)
        return prefix

    def _pull(self, prefix: str, dir_path: str) -> str:
        """Pulls a model to a destination"""
        try:
            logger.debug("Downloading from: %s...", prefix)
            destination = os.path.join(
                dir_path,
                os.path.split(prefix)[1],
            )
            blob = self.bucket.blob(prefix)
            blob.download_to_filename(destination)
            return destination
        except NotFound as exc:
            raise FilePullFailedException(exc) from exc

    def _remove(self, destination: str) -> bool:
        """Removes a file from the destination path"""
        if self.is_anon_client:
            raise NotImplementedError(
                "File removal is only supported for authenticated clients."
            )

        blob = self.bucket.blob(destination)
        if not blob.exists():
            logger.debug("Remote file does not exist: %s", destination)
            return False
        blob.delete()
        return True

    def _storage_location(self, prefix: str) -> metadata.Storage:
        """Returns a dict of the location the artifact was stored"""
        return metadata.Storage.from_bucket(
            storage_type="google:cloud-storage",
            bucket=self.bucket_name,
            prefix=prefix,
        )

    def _get_storage_location(self, meta_data: metadata.Storage) -> str:
        """Extracts the storage location from a meta data dictionary"""
        if self.bucket_name != meta_data.bucket:
            raise ValueError("Meta-data has a different bucket name")
        return meta_data.prefix

    def _read_json_objects(self, path: str) -> list:
        logger.debug("Listing files in: %s/%s", self.bucket_name, path)
        results = []
        blobs = self.client.list_blobs(
            self.bucket_name, prefix=path + "/", delimiter="/"
        )
        for blob in blobs:
            if not blob.name.endswith(".json"):
                continue
            obj = blob.download_as_string()
            try:
                obj = json.loads(obj)
                results.append(obj)
            except json.JSONDecodeError:
                continue
        return sorted_by_created(results)

    def _read_json_object(self, path: str) -> Optional[dict]:
        """Returns a dictionary of the JSON stored in a given path"""
        blob = self.bucket.blob(path)
        obj = blob.download_as_string()
        try:
            return json.loads(obj)
        except json.JSONDecodeError:
            return None
