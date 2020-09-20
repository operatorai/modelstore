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

from modelstore.clouds.storage import CloudStorage
from modelstore.clouds.util.paths import get_archive_path
from modelstore.clouds.util.versions import sorted_by_created
from modelstore.utils.log import logger

try:
    from google.auth.exceptions import DefaultCredentialsError
    from google.cloud import storage

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

    @classmethod
    def get_name(cls):
        return "google:cloud-storage"

    @property
    def client(self) -> "storage.Client":
        try:
            if self.__client is None:
                self.__client = storage.Client(self.project_name)
            return self.__client
        except DefaultCredentialsError:
            logger.error(
                "Missing credentials: https://cloud.google.com/docs/authentication/getting-started#command-line"  # noqa
            )
            raise

    def validate(self) -> bool:
        logger.debug("Querying for buckets with prefix=%s...", self.bucket_name)
        for bucket in list(self.client.list_buckets(prefix=self.bucket_name)):
            if bucket.name == self.bucket_name:
                return True
        return False

    def _push(self, source: str, destination: str) -> str:
        logger.info("Uploading to: %s...", destination)
        bucket = self.client.get_bucket(self.bucket_name)
        blob = bucket.blob(destination)
        blob.upload_from_filename(source)
        logger.debug("Finished: %s", destination)
        return destination

    def upload(self, domain: str, prefix: str, local_path: str) -> dict:
        bucket_path = get_archive_path(domain, prefix, local_path)
        return {
            "bucket": self.bucket_name,
            "prefix": self._push(local_path, bucket_path),
        }

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
