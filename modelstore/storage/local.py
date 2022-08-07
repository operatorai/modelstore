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
                raise Exception("Error: root_dir does not exist")
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

    def _get_metadata_path(
        self, domain: str, model_id: str, state_name: Optional[str] = None
    ) -> str:
        """Creates a path where a meta-data file about a model is stored.
        I.e.: :code:`operatorai-model-store/<domain>/versions/<model-id>.json`

        Args:
            domain (str): A group of models that are trained for the
            same end-use are given the same domain.

            model_id (str): A UUID4 string that identifies this specific
            model.
        """
        meta_data_path = super()._get_metadata_path(domain, model_id, state_name)
        return self.relative_dir(meta_data_path)

    def _push(self, source: str, destination: str) -> str:
        destination = self.relative_dir(destination)
        shutil.copy(source, destination)
        return destination

    def _pull(self, source: str, destination: str) -> str:
        if not os.path.exists(source):
            raise FilePullFailedException(f"File {source} does not exist.")

        try:
            file_name = os.path.split(source)[1]
            shutil.copy(source, destination)
            return os.path.join(os.path.abspath(destination), file_name)
        except FileNotFoundError as exc:
            raise FilePullFailedException(exc) from exc

    def _remove(self, destination: str) -> bool:
        """Removes a file from the destination path"""
        # @TODO: Empty directories are left behind after the destination file
        # has been deleted
        destination = self.relative_dir(destination)
        if not os.path.exists(destination):
            logger.debug("Remote file does not exist: %s", destination)
            return False
        os.remove(destination)
        return True

    def _read_json_objects(self, path: str) -> list:
        path = self.relative_dir(path)
        if not os.path.exists(path):
            return []
        results = []
        for entry in os.listdir(path):
            if not entry.endswith(".json"):
                continue
            version_path = os.path.join(path, entry)
            body = _read_json_file(version_path)
            if body is not None:
                results.append(body)
        return sorted_by_created(results)

    def relative_dir(self, file_path: str) -> str:
        """Returns the file_path, relative to the root_prefix for
        this local file system storage"""
        paths = os.path.split(file_path)
        parent_dir = os.path.join(self.root_prefix, paths[0])
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir)
        return os.path.join(parent_dir, paths[1])

    def _storage_location(self, prefix: str) -> metadata.Storage:
        """Returns a dict of the location the artifact was stored"""
        return metadata.Storage.from_path(
            storage_type="file_system",
            path=os.path.abspath(self.relative_dir(prefix))
        )

    def _get_storage_location(self, meta_data: metadata.Storage) -> str:
        """Extracts the storage location from a meta data dictionary"""
        return meta_data.path

    def _read_json_object(self, path: str) -> dict:
        path = self.relative_dir(path)
        return _read_json_file(path)


def _read_json_file(path: str) -> dict:
    try:
        with open(path, "r") as lines:
            return json.loads(lines.read())
    except json.JSONDecodeError:
        return None
