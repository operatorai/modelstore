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

from modelstore.storage.blob_storage import BlobStorage
from modelstore.storage.util.paths import (
    MODELSTORE_ROOT,
    get_model_state_path,
    is_valid_state_name,
)
from modelstore.storage.util.versions import sorted_by_created
from modelstore.utils.log import logger


class FileSystemStorage(BlobStorage):

    """
    File System Storage
    """

    def __init__(self, root_path: str):
        super().__init__([])
        if MODELSTORE_ROOT in root_path:
            warnings.warn(
                f'Warning: "{MODELSTORE_ROOT}" is in the root path, and is a value'
                + " that this library usually appends. Is this intended?"
            )
        root_path = os.path.abspath(root_path)
        self.root_dir = root_path
        logger.debug("Root is: %s", self.root_dir)

    def validate(self) -> bool:
        """This validates that the directory exists
        and can be written to"""
        # pylint: disable=broad-except
        try:
            parent_dir = os.path.split(self.root_dir)[0]
            # Check that directory exists
            if not os.path.exists(parent_dir):
                logger.error("Error: %s does not exist", parent_dir)
                return False

            # Check we can write to it
            source = os.path.join(self.root_dir, ".operator-ai")
            Path(source).touch()
            os.remove(source)
            return True
        except Exception as ex:
            logger.error("Error=%s...", str(ex))
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
        meta_data_path = super()._get_metadata_path(
            domain, model_id, state_name
        )
        return self.relative_dir(meta_data_path)

    def _push(self, source: str, destination: str) -> str:
        destination = self.relative_dir(destination)

        shutil.copy(source, destination)
        return destination

    def _pull(self, source: str, destination: str) -> str:
        file_name = os.path.split(source)[1]
        shutil.copy(source, destination)
        return os.path.join(os.path.abspath(destination), file_name)

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
        paths = os.path.split(file_path)
        parent_dir = os.path.join(self.root_dir, paths[0])
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir)
        return os.path.join(parent_dir, paths[1])

    def _storage_location(self, prefix: str) -> dict:
        """ Returns a dict of the location the artifact was stored """
        return {
            "type": "file_system",
            "path": os.path.abspath(prefix),
        }

    def _get_storage_location(self, meta: dict) -> str:
        """ Extracts the storage location from a meta data dictionary """
        return meta["path"]

    def _read_json_object(self, path: str) -> dict:
        path = self.relative_dir(path)
        return _read_json_file(path)

    def state_exists(self, state_name: str) -> bool:
        """ Returns whether a model state with name state_name exists """
        if not is_valid_state_name(state_name):
            return False
        # @TODO this function can be removed once get_model_state_path
        # doesn't need to be called with relative_dir()
        state_path = self.relative_dir(get_model_state_path(state_name))
        return os.path.exists(state_path)


def _read_json_file(path: str) -> dict:
    try:
        with open(path, "r") as lines:
            return json.loads(lines.read())
    except json.JSONDecodeError:
        return None
