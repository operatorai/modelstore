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

from modelstore.storage.storage import CloudStorage
from modelstore.storage.util.paths import get_archive_path
from modelstore.storage.util.versions import sorted_by_created
from modelstore.utils.log import logger

# pylint: disable=protected-access


class OperatorStorage(CloudStorage):

    """
    OperatorStorage is a managed model store. No dependencies required.

    Usage of this storage requires you to have an `api_key`.
    """

    def __init__(self, api_key: str):
        super().__init__([])
        self.api_key = api_key

    def validate(self) -> bool:
        """ No dependencies or setup required; validation returns True """
        # @TODO add validation check
        return len(self.api_key) > 0

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
        # @TODO request a presigned URL
        # @TODO use curl to upload the model
        return {
            "type": "operator:cloud-storage",
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

    def _read_json_object(self, path: str) -> dict:
        """ Returns a dictionary of the JSON stored in a given path """
        bucket = self.client.get_bucket(self.bucket_name)
        blob = bucket.blob(path)
        obj = blob.download_as_string()
        return json.loads(obj)
