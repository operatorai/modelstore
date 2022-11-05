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
import shutil
import warnings
from pathlib import Path
from typing import Optional

from modelstore.metadata import metadata
from modelstore.storage.blob_storage import BlobStorage
from modelstore.storage.util.paths import (
    MODELSTORE_ROOT_PREFIX,
)
from modelstore.storage.util.versions import sorted_by_created
from modelstore.utils.log import logger
from modelstore.utils.exceptions import FilePullFailedException


class FileSystemStorage(BlobStorage):

    """
    File System Storage: store models in a directory
    """

    NAME = "filesystem"
    BUILD_FROM_ENVIRONMENT = {
        "required": [
            "MODEL_STORE_ROOT_PREFIX",
        ],
        "optional": [],
    }

    def __init__(self, root_dir: Optional[str] = None, create_directory: bool = False):
        super().__init__(
            required_deps=[],
            root_prefix=root_dir,
            root_prefix_env_key="MODEL_STORE_ROOT_PREFIX"
        )
        if self.root_prefix == "":
            raise Exception(
                "Error: cannot create a file system model store without a root directory"
            )
        if MODELSTORE_ROOT_PREFIX in self.root_prefix:
            warnings.warn(
                f'Warning: "{MODELSTORE_ROOT_PREFIX}" is in the root path, and is a value'
                + " that this library usually appends. Is this intended?"
            )
        self.root_prefix = os.path.abspath(self.root_prefix)
        self._create_directory = create_directory

    def validate(self) -> bool:
        """This validates that the directory exists and can be written to"""
        # pylint: disable=broad-except
        # Check that the directory exists & we can write to it
        parent_dir = os.path.split(self.root_prefix)[0]
        if not os.path.exists(parent_dir):
            raise Exception(
                f"Error: Parent directory to root dir '{parent_dir}' does not exist"
            )

        if not os.path.exists(self.root_prefix):
            if not self._create_directory:
                raise FileNotFoundError("Error: root_dir does not exist")
            logger.debug("creating root directory %s", self.root_prefix)
            os.mkdir(self.root_prefix)
        if not os.path.isdir(self.root_prefix):
            raise Exception("Error: root_dir needs to be a directory")

        try:
            # Check we can write to it
            source = os.path.join(self.root_prefix, ".operator-ai")
            Path(source).touch()
            os.remove(source)
            return True
        except Exception as ex:
            logger.error(ex)
            return False

    def _storage_path(self, prefix: str, create_parent: bool = False) -> str:
        storage_path = os.path.join(
            self.root_prefix,
            prefix,
        )
        if create_parent:
            # Create the parent directories if they are missing
            parent_dir = os.path.split(storage_path)[0]
            if not os.path.exists(parent_dir):
                logger.debug("Creating: %s", parent_dir)
                os.makedirs(parent_dir)
        return storage_path

    def _get_storage_location(self, meta_data: metadata.Storage) -> str:
        """Extracts the storage location from a meta data dictionary"""
        return self._storage_path(meta_data.path, create_parent=False)

    def _push(self, source: str, destination: str) -> str:
        """ Copies a file with path at source to <root>/<destination> """
        destination_prefix = os.path.join(
            destination,
            os.path.split(source)[1],
        )
        destination_path = self._storage_path(
            destination_prefix,
            create_parent=True,
        )
        shutil.copy(source, destination_path)
        return destination_prefix

    def _pull(self, source: str, destination: str) -> str:
        """ Copies a file from destination to source """
        source_path = self._storage_path(
            source,
            create_parent=False,
        )
        destination_path = os.path.join(
            destination,
            os.path.split(source_path)[1],
        )
        try:
            shutil.copy(source_path, destination_path)
            return destination_path
        except FileNotFoundError as exc:
            logger.error(exc)
            raise FilePullFailedException(exc) from exc

    def _remove(self, destination: str) -> bool:
        """ Removes a file from the destination path
        and any empty directories up the tree """
        destination_path = self._storage_path(
            destination,
            create_parent=False,
        )
        if not os.path.exists(destination_path):
            logger.debug("Remote file does not exist: %s", destination)
            return False
        os.remove(destination_path)

        parent_dir = os.path.split(destination_path)[0]
        while len(os.listdir(parent_dir)) == 0:
            try:
                logger.debug("Removing directory: %s", parent_dir)
                os.rmdir(parent_dir)
                parent_dir = os.path.split(parent_dir)[0]
                if parent_dir == self.root_prefix:
                    break
            except OSError:
                break
        return True

    def _read_json_objects(self, path: str) -> list:
        source_path = self._storage_path(
            path,
            create_parent=False,
        )
        if not os.path.exists(source_path):
            return []
        results = []
        for entry in os.listdir(source_path):
            if not entry.endswith(".json"):
                continue
            version_path = os.path.join(source_path, entry)
            body = self._read_json_object(version_path)
            if body is not None:
                results.append(body)
        return sorted_by_created(results)

    def _storage_location(self, prefix: str) -> metadata.Storage:
        """Returns a dict of the location the artifact was stored"""
        return metadata.Storage.from_path(
            storage_type="file_system",
            root=self.root_prefix,
            path=prefix,
        )

    def _read_json_object(self, path: str) -> dict:
        source_path = self._storage_path(
            path,
            create_parent=False,
        )
        try:
            with open(source_path, "r") as lines:
                return json.loads(lines.read())
        except json.JSONDecodeError:
            return None
