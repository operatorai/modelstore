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

from modelstore.storage.blob_storage import BlobStorage
from modelstore.storage.util.paths import _ROOT, get_archive_path
from modelstore.storage.util.versions import sorted_by_created
from modelstore.utils.log import logger


class FileSystemStorage(BlobStorage):

    """
    File System Storage
    """

    def __init__(self, root_path: str):
        super().__init__([])
        if _ROOT in root_path:
            warnings.warn(
                f'Warning: "{_ROOT}" is in the root path, and is a value'
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

    def _push(self, source: str, destination: str) -> str:
        destination = self.relative_dir(destination)
        shutil.copy(source, destination)
        return destination

    def _pull(self, source: dict, destination: str) -> str:
        path = _get_location(source)
        file_name = os.path.split(path)[1]
        shutil.copy(path, destination)
        return os.path.join(os.path.abspath(destination), file_name)

    def upload(self, domain: str, model_id: str, local_path: str) -> dict:
        fs_path = get_archive_path(domain, local_path)
        logger.info("Moving to: %s...", fs_path)
        archive_path = self._push(local_path, fs_path)
        logger.debug("Finished: %s", fs_path)
        return _format_location(archive_path)

    def _read_json_objects(self, path: str) -> list:
        results = []
        path = self.relative_dir(path)
        for entry in os.listdir(path):
            if not entry.endswith(".json"):
                # @TODO tighter controls
                continue
            version_path = os.path.join(path, entry)
            meta = _read_json_file(version_path)
            results.append(meta)
        return sorted_by_created(results)

    def _read_json_object(self, path: str) -> dict:
        path = self.relative_dir(path)
        return _read_json_file(path)

    def relative_dir(self, file_path: str) -> str:
        paths = os.path.split(file_path)
        parent_dir = os.path.join(self.root_dir, paths[0])
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir)
        return os.path.join(parent_dir, paths[1])


def _format_location(archive_path: str) -> dict:
    return {
        "type": "file_system",
        "path": os.path.abspath(archive_path),
    }


def _get_location(meta: dict) -> str:
    return meta["path"]


def _read_json_file(path: str) -> dict:
    with open(path, "r") as lines:
        meta = json.loads(lines.read())
    return meta
