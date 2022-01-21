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
import tempfile
from abc import ABCMeta, abstractmethod
from datetime import datetime
from typing import Optional, Union

from modelstore.storage.storage import CloudStorage
from modelstore.storage.util import environment
from modelstore.storage.util.paths import (
    get_archive_path,
    get_domain_path,
    get_domains_path,
    get_model_state_path,
    get_versions_path,
    is_valid_state_name,
)
from modelstore.utils.log import logger


class BlobStorage(CloudStorage):

    """
    Abstract class capturing a file system type of cloud storage
    (e.g., Google Cloud Storage, AWS S3, local file system)
    """

    __metaclass__ = ABCMeta

    def __init__(self, required_deps: list, root_prefix: str = None):
        super().__init__(required_deps)
        root_prefix = environment.get_value(
            root_prefix, "MODEL_STORE_ROOT_PREFIX", allow_missing=True
        )
        self.root_prefix = root_prefix if root_prefix is not None else ""
        logger.debug("Root prefix is: %s", self.root_prefix)

    @abstractmethod
    def _push(self, source: str, destination: str) -> str:
        """ Pushes a file from a source to a destination """
        raise NotImplementedError()

    @abstractmethod
    def _pull(self, source: str, destination: str) -> str:
        """ Pulls a model from a source to a destination """
        raise NotImplementedError()

    @abstractmethod
    def _remove(self, destination: str) -> bool:
        """ Removes a file from the destination path """
        raise NotImplementedError()

    @abstractmethod
    def _read_json_objects(self, path: str) -> list:
        """ Returns a list of all the JSON in a path """
        raise NotImplementedError()

    @abstractmethod
    def _read_json_object(self, path: str) -> dict:
        """ Returns a dictionary of the JSON stored in a given path """
        raise NotImplementedError()

    @abstractmethod
    def _storage_location(self, prefix: str) -> dict:
        """ Returns a dict of the location the artifact was stored """
        raise NotImplementedError()

    @abstractmethod
    def _get_storage_location(self, meta: dict) -> str:
        """ Extracts the storage location from a meta data dictionary """
        raise NotImplementedError()

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
        versions_path = get_versions_path(self.root_prefix, domain, state_name)
        return os.path.join(versions_path, f"{model_id}.json")

    def _upload_extra(self, local_path: str, remote_path: str):
        if os.path.isdir(local_path):
            #  Currently ignoring directories
            return
        local_file_name = os.path.split(local_path)[1]
        if local_file_name == "artifacts.tar.gz":
            raise ValueError("Name conflict in extras: cannot use 'artifacts.tar.gz'")
        remote_file_path = os.path.join(remote_path, local_file_name)
        self._push(local_path, remote_file_path)

    def upload(
        self,
        domain: str,
        local_path: str,
        extras: Optional[Union[str, list]] = None,
    ) -> dict:
        # Upload the archive into storage
        archive_remote_path = get_archive_path(self.root_prefix, domain, local_path)
        prefix = self._push(local_path, archive_remote_path)
        if extras:
            # If any extras have been defined, they are uploaded
            # to the same place
            remote_path = os.path.split(archive_remote_path)[0]
            if isinstance(extras, list):
                for extra_path in extras:
                    self._upload_extra(extra_path, remote_path)
            else:
                self._upload_extra(extras, remote_path)

        return self._storage_location(prefix)

    def set_meta_data(self, domain: str, model_id: str, meta_data: dict):
        logger.debug("Copying meta-data: %s", meta_data)
        with tempfile.TemporaryDirectory() as tmp_dir:
            local_path = os.path.join(tmp_dir, f"{model_id}.json")
            with open(local_path, "w") as out:
                out.write(json.dumps(meta_data))
            self._push(local_path, self._get_metadata_path(domain, model_id))
            self._push(local_path, get_domain_path(self.root_prefix, domain))

    def get_meta_data(self, domain: str, model_id: str) -> dict:
        """ Returns a model's meta data """
        if any(x in [None, ""] for x in [domain, model_id]):
            raise ValueError("domain and model_id must be set")
        remote_path = self._get_metadata_path(domain, model_id)
        with tempfile.TemporaryDirectory() as tmp_dir:
            local_path = self._pull(remote_path, tmp_dir)
            with open(local_path, "r") as lines:
                return json.loads(lines.read())

    def download(self, local_path: str, domain: str, model_id: str = None):
        """Downloads an artifacts archive for a given (domain, model_id) pair.
        If no model_id is given, it defaults to the latest model in that
        domain"""
        model_meta = None
        if model_id is None:
            model_domain = get_domain_path(self.root_prefix, domain)
            model_meta = self._read_json_object(model_domain)
            logger.info("Latest model is: %s", model_meta["model"]["model_id"])
        else:
            model_meta_path = self._get_metadata_path(domain, model_id)
            # Note: this will fail if the model does not exist (needs a more informative exception)
            model_meta = self._read_json_object(model_meta_path)
        storage_path = self._get_storage_location(model_meta["storage"])
        return self._pull(storage_path, local_path)

    def list_domains(self) -> list:
        """ Returns a list of all the existing model domains """
        domains = get_domains_path(self.root_prefix)
        domains = self._read_json_objects(domains)
        return [d["model"]["domain"] for d in domains]

    def list_versions(self, domain: str, state_name: Optional[str] = None) -> list:
        if state_name and not self.state_exists(state_name):
            raise Exception(f"State: '{state_name}' does not exist")
        versions_path = get_versions_path(self.root_prefix, domain, state_name)
        versions = self._read_json_objects(versions_path)
        # @TODO sort models by creation time stamp
        return [v["model"]["model_id"] for v in versions]

    def state_exists(self, state_name: str) -> bool:
        """ Returns whether a model state with name state_name exists """
        if not is_valid_state_name(state_name):
            return False
        try:
            state_path = get_model_state_path(self.root_prefix, state_name)
            with tempfile.TemporaryDirectory() as tmp_dir:
                self._pull(state_path, tmp_dir)
            return True
            # pylint: disable=broad-except,invalid-name
        except Exception as e:
            # @TODO - check the error type
            logger.error("Error checking state: %s", str(e))
            return False

    def create_model_state(self, state_name: str):
        """ Creates a state label that can be used to tag models """
        if not is_valid_state_name(state_name):
            raise Exception(f"Cannot create state with name: '{state_name}'")
        if self.state_exists(state_name):
            logger.info("Model state '%s' already exists", state_name)
            return
        logger.debug("Creating model state: %s", state_name)
        with tempfile.TemporaryDirectory() as tmp_dir:
            version_path = os.path.join(tmp_dir, f"{state_name}.json")
            with open(version_path, "w") as out:
                state_data = {
                    "created": datetime.now().strftime("%Y/%m/%d/%H:%M:%S"),
                    "state_name": state_name,
                }
                out.write(json.dumps(state_data))
            self._push(version_path, get_model_state_path(self.root_prefix, state_name))

    def set_model_state(self, domain: str, model_id: str, state_name: str):
        """ Adds the given model ID to the set that are in the state_name path """
        if not self.state_exists(state_name):
            logger.debug("Model state '%s' does not exist", state_name)
            raise ValueError(f"State '{state_name}' does not exist")
        model_path = self._get_metadata_path(domain, model_id)
        model_state_path = self._get_metadata_path(domain, model_id, state_name)
        with tempfile.TemporaryDirectory() as tmp_dir:
            local_model_path = self._pull(model_path, tmp_dir)
            self._push(local_model_path, model_state_path)
        logger.debug("Successfully set %s=%s from %s", domain, model_id, state_name)

    def unset_model_state(self, domain: str, model_id: str, state_name: str):
        """ Removes the given model ID from the set that are in the state_name path """
        if not self.state_exists(state_name):
            logger.debug("Model state '%s' does not exist", state_name)
            raise ValueError(f"State '{state_name}' does not exist")
        model_state_path = self._get_metadata_path(domain, model_id, state_name)
        if self._remove(model_state_path):
            logger.debug(
                "Successfully unset %s=%s from %s", domain, model_id, state_name
            )
