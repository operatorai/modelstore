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
        super().__init__(["pydoop"], root_prefix, "MODEL_STORE_HDFS_ROOT_PREFIX")
        logger.debug("creating root directory %s", create_directory)
        self._create_directory = create_directory

    def validate(self) -> bool:
        try:
            hdfs.ls(self.root_prefix)
            return True
        except FileNotFoundError as exc:
            if not self._create_directory:
                logger.exception(exc)
                return False
            logger.debug("creating root directory %s", self.root_prefix)
            hdfs.mkdir(self.root_prefix)
            return True

    def _push(self, file_path: str, prefix: str) -> str:
        logger.info("Uploading to: %s...", prefix)
        # This will raise an exception if the file already exists
        hdfs.mkdir(os.path.split(prefix)[0])
        if hdfs.path.exists(prefix):
            hdfs.rm(prefix)
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
        if hdfs.path.exists(prefix):
            logger.debug("Deleting: %s...", prefix)
            hdfs.rm(prefix)
            return True
        return False

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
        if not hdfs.path.exists(prefix):
            return results
        for obj in hdfs.ls(prefix):
            logger.debug("reading: %s", obj)
            if not hdfs.path.basename(obj).endswith(".json"):
                logger.debug("Skipping non-json file: %s", obj)
                continue
            parent = obj[obj.index(prefix):]
            if os.path.split(parent)[0] != prefix:
                # We don't want to read files in a sub-prefix
                logger.debug("Skipping file in sub-prefix: %s", obj)
                continue
            json_obj = self._read_json_object(obj)
            if json_obj is not None:
                results.append(json_obj)
        return sorted_by_created(results)

    def _read_json_object(self, prefix: str) -> dict:
        logger.debug("Reading: %s", prefix)
        lines = hdfs.load(prefix)
        if len(lines) == 0:
            return None
        try:
            return json.loads(lines)
        except json.JSONDecodeError as exc:
            logger.exception(exc)
            return None
