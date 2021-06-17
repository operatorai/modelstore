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
from typing import Optional

from modelstore.storage.storage import CloudStorage
from modelstore.storage.util.paths import (
    get_domain_path,
    get_domains_path,
    get_metadata_path,
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

    @abstractmethod
    def _push(self, source: str, destination: str) -> str:
        """ Pushes a file from a source to a destination """
        raise NotImplementedError()

    @abstractmethod
    def _pull(self, source: dict, destination: str) -> str:
        """ Pulls a model from a source to a destination """
        raise NotImplementedError()

    @abstractmethod
    def _read_json_objects(self, path: str) -> list:
        """ Returns a list of all the JSON in a path """
        raise NotImplementedError()

    @abstractmethod
    def _read_json_object(self, path: str) -> dict:
        """ Returns a dictionary of the JSON stored in a given path """
        raise NotImplementedError()

    def _state_exists(self, state_name: str) -> bool:
        """ Returns whether a model state with name state_name exists """
        # @TODO implement
        if not is_valid_state_name(state_name):
            return False
        return False

    def set_meta_data(self, domain: str, model_id: str, meta_data: dict):
        logger.debug("Copying meta-data: %s", meta_data)
        with tempfile.TemporaryDirectory() as tmp_dir:
            version_path = os.path.join(tmp_dir, f"{model_id}.json")
            with open(version_path, "w") as out:
                out.write(json.dumps(meta_data))
            self._push(version_path, get_metadata_path(domain, model_id))
            self._push(version_path, get_domain_path(domain))

    def download(self, local_path: str, domain: str, model_id: str = None):
        """Downloads an artifacts archive for a given (domain, model_id) pair.
        If no model_id is given, it defaults to the latest model in that
        domain"""
        model_meta = None
        if model_id is None:
            model_domain = get_domain_path(domain)
            model_meta = self._read_json_object(model_domain)
            logger.info("Latest model is: %f", model_meta["model"]["model_id"])
        else:
            model_meta_path = get_metadata_path(domain, model_id)
            model_meta = self._read_json_object(model_meta_path)
        return self._pull(model_meta["storage"], local_path)

    def list_domains(self) -> list:
        """ Returns a list of all the existing model domains """
        domains = get_domains_path()
        domains = self._read_json_objects(domains)
        return [d["model"]["domain"] for d in domains]

    def list_versions(
        self, domain: str, state_name: Optional[str] = None
    ) -> list:
        if state_name and not self._state_exists(state_name):
            raise Exception(f"State: '{state_name}' does not exist")
        versions_path = get_versions_path(domain, state_name)
        versions = self._read_json_objects(versions_path)
        return [v["model"]["model_id"] for v in versions]

    def create_model_state(self, state_name: str):
        """ Creates a state label that can be used to tag models """
        if not is_valid_state_name(state_name):
            raise Exception(f"Cannot create state with name: '{state_name}'")
        if self._state_exists(state_name):
            logger.debug("Model state '%s' already exists", state_name)
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
            self._push(version_path, get_model_state_path(state_name))

    def set_model_state(self, domain: str, model_id: str, state_name: str):
        """ Adds the given model ID the set that are in the state_name path """
        if not self._state_exists(state_name):
            logger.debug("Model state '%s' does not exist", state_name)
            raise Exception(f"State '{state_name}' does not exist")
        model_meta_data_path = get_metadata_path(domain, model_id)
        versions_path = get_versions_path(domain, state_name)
        with tempfile.TemporaryDirectory() as tmp_dir:
            meta_data = self._pull(model_meta_data_path, tmp_dir)
            self._push(meta_data, versions_path)
