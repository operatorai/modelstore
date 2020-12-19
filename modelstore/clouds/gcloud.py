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

from modelstore.clouds.storage import CloudStorage
from modelstore.clouds.util.paths import get_archive_path
from modelstore.clouds.util.versions import sorted_by_created
from modelstore.utils.log import logger

# pylint: disable=protected-access

try:
    from google.auth.exceptions import DefaultCredentialsError
    from google.cloud import storage

    storage.blob._DEFAULT_CHUNKSIZE = 2097152  # 1024 * 1024 B * 2 = 2 MB
    storage.blob._MAX_MULTIPART_SIZE = 2097152  # 2 MB

    GCLOUD_EXISTS = True
except ImportError:
    GCLOUD_EXISTS = False


class GoogleCloudStorage(CloudStorage):

    """
    Google Cloud Storage
    """

    def __init__(
        self,
        project_name: str,
        bucket_name: str,
        client: "storage.Client" = None,
    ):
        super().__init__(["google.cloud.storage"])
        self.project_name = project_name
        self.bucket_name = bucket_name
        self.__client = client

    @property
    def client(self) -> "storage.Client":
        if not GCLOUD_EXISTS:
            raise ImportError("Please install google-cloud-storage")
        try:
            if self.__client is None:
                self.__client = storage.Client(self.project_name)
            return self.__client
        except DefaultCredentialsError:
            try:
                # Try to authenticate, if in a Colab notebook
                from google.colab import auth

                auth.authenticate_user()
                self.__client = storage.Client(self.project_name)
                return self.__client
            except ModuleNotFoundError:
                pass
            logger.error(
                "Missing credentials: https://cloud.google.com/docs/authentication/getting-started#command-line"  # noqa
            )
            raise

    def validate(self) -> bool:
        """Runs any required validation steps - e.g.,
        checking that a cloud bucket exists"""
        logger.debug("Querying for buckets with prefix=%s...", self.bucket_name)
        for bucket in list(self.client.list_buckets(prefix=self.bucket_name)):
            if bucket.name == self.bucket_name:
                return True
        return False

    def _push(self, source: str, destination: str) -> str:
        logger.info("Uploading to: %s...", destination)
        bucket = self.client.get_bucket(self.bucket_name)
        blob = bucket.blob(destination)

        ## For slow upload speed
        # https://stackoverflow.com/questions/61001454/why-does-upload-from-file-google-cloud-storage-function-throws-timeout-error

        with open(source, "rb") as f:
            blob.upload_from_file(f)
        logger.debug("Finished: %s", destination)
        return destination

    def _pull(self, source: dict, destination: str) -> str:
        """ Pulls a model to a destination """
        logger.info("Downloading from: %s...", source)
        prefix = _get_location(self.bucket_name, source)
        file_name = os.path.split(prefix)[1]
        destination = os.path.join(destination, file_name)
        bucket = self.client.get_bucket(self.bucket_name)
        blob = bucket.blob(prefix)
        blob.download_to_filename(destination)
        logger.debug("Finished: %s", destination)
        return destination

    def upload(self, domain: str, local_path: str) -> dict:
        bucket_path = get_archive_path(domain, local_path)
        prefix = self._push(local_path, bucket_path)
        return _format_location(self.bucket_name, prefix)

    def _read_json_objects(self, path: str) -> list:
        results = []
        blobs = self.client.list_blobs(
            self.bucket_name, prefix=path + "/", delimiter="/"
        )
        for blob in blobs:
            if not blob.name.endswith(".json"):
                # @TODO tighter controls here
                continue
            obj = blob.download_as_string()
            results.append(json.loads(obj))
        return sorted_by_created(results)

    def _read_json_object(self, path: str) -> dict:
        """ Returns a dictionary of the JSON stored in a given path """
        bucket = self.client.get_bucket(self.bucket_name)
        blob = bucket.blob(path)
        obj = blob.download_as_string()
        return json.loads(obj)


def _format_location(bucket_name: str, prefix: str) -> dict:
    return {
        "type": "google:cloud-storage",
        "bucket": bucket_name,
        "prefix": prefix,
    }


def _get_location(bucket_name, meta: dict) -> str:
    if bucket_name != meta["bucket"]:
        raise ValueError("Meta-data has a different bucket name")
    return meta["prefix"]
