#    Copyright 2023 Neal Lathia
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
from modelstore.storage.util.versions import sorted_by_created
from modelstore.utils.log import logger
from modelstore.utils.exceptions import FilePullFailedException

try:
    import pydoop.hdfs as hdfs

    HDFS_EXISTS = True
except ImportError:
    HDFS_EXISTS = False


class HdfsStorage(BlobStorage):

    """
    HDFS Storage

    Assumes that you have `pydoop` installed
    https://crs4.github.io/pydoop/tutorial/hdfs_api.html#hdfs-api-tutorial
    """

    NAME = "hdfs"
    BUILD_FROM_ENVIRONMENT = {
        "required": [],
        "optional": [
            "MODEL_STORE_HDFS_ROOT_PREFIX",
        ],
    }

    def __init__(self, root_prefix: Optional[str] = None, create_directory: bool = False):
        super().__init__(["hdfs"], root_prefix, "MODEL_STORE_HDFS_ROOT_PREFIX")
        self._create_directory = create_directory

    def validate(self) -> bool:
        logger.debug("Creating path=%s...", self.root_prefix)
        # @TODO check if root_prefix already exists
        hdfs.mkdir(self.root_prefix)
        return True

    def _push(self, file_path: str, prefix: str) -> str:
        logger.info("Uploading to: %s...", prefix)
        hdfs.put(file_path, prefix)
        return prefix

    def _pull(self, prefix: str, dir_path: str) -> str:
        try:
            logger.debug("Downloading from: %s...", prefix)
            file_name = os.path.split(prefix)[1]
            destination = os.path.join(dir_path, file_name)
            hdfs.get(prefix, destination)
            return destination
        except Exception as exc:
            raise FilePullFailedException(exc) from exc

    def _remove(self, prefix: str) -> bool:
        """Removes a file from the destination path"""
        logger.debug("Deleting: %s...", prefix)
        hdfs.rm(prefix)
        return True

    def _storage_location(self, prefix: str) -> metadata.Storage:
        """Returns a dict of the location the artifact was stored"""
        return metadata.Storage.from_path(
            storage_type="hdfs",
            root=self.root_prefix,
            path=prefix,
        )

    def _get_storage_location(self, meta_data: metadata.Storage) -> str:
        """Extracts the storage location from a meta data dictionary"""
        return meta_data.path

    def _read_json_objects(self, prefix: str) -> list:
        logger.debug("Listing files in: %s", prefix)
        results = []
        objects = [f for f in hdfs.ls(os.path.join(prefix))]
        for obj in objects:
            if not hdfs.path.basename(obj).endswith(".json"):
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
        logger.debug("Reading: %s", prefix)
        obj = hdfs.load(prefix)
        lines = obj.readlines()
        if len(lines) == 0:
            return None
        try:
            return json.loads(lines[0])
        except json.JSONDecodeError:
            return None
