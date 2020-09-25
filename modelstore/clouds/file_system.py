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

from modelstore.clouds.storage import CloudStorage
from modelstore.clouds.util.paths import _ROOT, get_archive_path
from modelstore.clouds.util.versions import sorted_by_created
from modelstore.utils.log import logger


class FileSystemStorage(CloudStorage):

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
        logger.info("Root is: %s", self.root_dir)

    @classmethod
    def get_name(cls):
        return "file_system"

    def relative_dir(self, file_path: str) -> str:
        paths = os.path.split(file_path)
        parent_dir = os.path.join(self.root_dir, paths[0])
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir)
        return os.path.join(parent_dir, paths[1])

    def validate(self) -> bool:
        """ This validates that the directory exists """
        # pylint: disable=broad-except
        try:
            parent_dir = os.path.split(self.root_dir)[0]
            # Check that directory exists
            if not os.path.exists(parent_dir):
                logger.error("Error: %s does not exist", parent_dir)
                return False

            # Check we can write to it
            source = os.path.join(parent_dir, ".operator-ai")
            Path(source).touch()
            os.remove(source)
            return True
        except Exception as ex:
            logger.error("Error=%s...", str(ex))
            return False

    def upload(self, domain: str, prefix: str, local_path: str) -> dict:
        fs_path = get_archive_path(domain, prefix, local_path)
        logger.info("Moving to: %s...", fs_path)
        archive_path = self._push(local_path, fs_path)
        logger.debug("Finished: %s", fs_path)
        return {"path": os.path.abspath(archive_path)}

    def _push(self, source: str, destination: str) -> str:
        """ Pushes a file to a destination """
        destination = self.relative_dir(destination)
        shutil.copy(source, destination)
        return destination

    def _read_json_objects(self, path: str) -> list:
        results = []
        path = self.relative_dir(path)
        for entry in os.listdir(path):
            if not entry.endswith(".json"):
                # @TODO tighter controls
                continue
            version_path = os.path.join(path, entry)
            with open(version_path, "r") as lines:
                meta = json.loads(lines.read())
                results.append(meta)
        return sorted_by_created(results)
