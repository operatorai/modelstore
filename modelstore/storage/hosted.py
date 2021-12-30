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
from typing import Optional

import requests
from modelstore.storage.storage import CloudStorage
from modelstore.utils.log import logger
from tqdm import tqdm
from tqdm.utils import CallbackIOWrapper

# pylint: disable=protected-access

_URL_ROOT = "https://k8evynth8b.execute-api.eu-west-1.amazonaws.com/prod/"


class HostedStorage(CloudStorage):

    """
    HostedStorage is a managed model store.
    Usage of this storage requires you to have an `api_access_key` and `api_key_id`.
    """

    NAME = "hosted"
    BUILD_FROM_ENVIRONMENT = {
        "required": ["MODELSTORE_KEY_ID", "MODELSTORE_ACCESS_KEY"],
        "optional": [],
    }

    def __init__(self, access_key_id: Optional[str], secret_access_key: Optional[str]):
        super().__init__([])
        self.access_key_id = _get_environ(access_key_id, "MODELSTORE_KEY_ID")
        self.secret_access_key = _get_environ(
            secret_access_key, "MODELSTORE_ACCESS_KEY"
        )

    def validate(self) -> bool:
        """ Requires an ACCESS_KEY_ID and SECRET_ACCESS_KEY """
        for key in [self.access_key_id, self.secret_access_key]:
            if key is None or len(key) == 0:
                return False
        try:
            _ = self._post("ping", {})
            return True
        except Exception:
            return False

    def _post(self, endpoint: str, data: dict) -> dict:
        url = os.path.join(_URL_ROOT, endpoint)
        logger.info("ℹ️  POST: %s", url)
        headers = {"x-api-key": self.secret_access_key}
        data["key_id"] = self.access_key_id
        rsp = requests.post(url, headers=headers, data=json.dumps(data))
        if rsp.status_code != 200:
            logger.debug("Request failed: %s", rsp.status_code)
            raise Exception(f"Error: {rsp.text.strip()}")
        return rsp.json()

    def _get_url(self, endpoint: str, domain: str, model_id: str) -> str:
        """ Returns a pre-signed URL for up/downloading models """
        data = {"domain": domain, "model_id": model_id}
        rsp = self._post(endpoint, data)
        return rsp["url"]

    def _register_upload(self, domain: str, model_id: str):
        data = {"domain": domain, "model_id": model_id}
        self._post("uploaded", data)

    def upload(self, domain: str, model_id: str, local_path: str) -> dict:
        upload_url = self._get_url("upload-url", domain, model_id)
        _upload(local_path, upload_url)
        self._register_upload(domain, model_id)
        return {
            "type": "operator:cloud-storage",
        }

    def list_versions(self, domain: str, state_name: Optional[str] = None) -> list:
        """ Returns a list of a model's versions """
        # @TODO enable filtering by state-name
        rsp = self._post("list-models", {"domain": domain})
        return rsp["models"]

    def list_domains(self) -> list:
        """ Returns a list of all the existing model domains """
        rsp = self._post("list-domains", {})
        return rsp["domains"]

    def set_meta_data(self, domain: str, model_id: str, meta_data: dict):
        """ Annotates a model with some given meta data """
        data = {
            "domain": domain,
            "model_id": model_id,
            "meta_data": json.dumps(meta_data),
        }
        self._post("set-metadata", data)

    def get_meta_data(self, domain: str, model_id: str) -> dict:
        """ Returns a model's meta data """
        # @TODO Future
        raise NotImplementedError()

    def download(self, local_path: str, domain: str, model_id: str = None):
        """Downloads an artifacts archive for a given (domain, model_id) pair.
        If no model_id is given, it defaults to the latest model in that
        domain"""
        download_url = self._get_url("download-url", domain, model_id)
        return _download(local_path, download_url)

    def create_model_state(self, state_name: str):
        """ Creates a state label that can be used to tag models """
        # @TODO Future
        raise NotImplementedError()

    def set_model_state(self, domain: str, model_id: str, state_name: str):
        """ Adds the given model ID the set that are in the state_name path """
        # @TODO Future
        raise NotImplementedError()

    def unset_model_state(self, domain: str, model_id: str, state_name: str):
        """ Removes the given model ID from the set that are in the state_name path """
        # @TODO Future
        raise NotImplementedError()


def _get_environ(value: str, key: str) -> str:
    if value is not None:
        return value
    return os.environ.get(key)


def _upload(local_path, upload_url):
    """ Uploads a local file to an upload URL, showing a progress bar """
    file_path = os.path.abspath(local_path)
    file_size = os.stat(file_path).st_size
    with open(file_path, "rb") as f:
        with tqdm(
            total=file_size, unit="B", unit_scale=True, unit_divisor=1024
        ) as progress:
            wrapped_file = CallbackIOWrapper(progress.update, f, "read")
            requests.put(upload_url, data=wrapped_file)


def _download(local_path, download_url) -> str:
    """ Downloads a remove file from a URL, showing a progress bar """
    resp = requests.get(download_url, stream=True)
    total_length = int(resp.headers.get("content-length", 0))
    logger.info(total_length)
    archive_file = os.path.join(local_path, "artifacts.tar.gz")
    with open(archive_file, "wb") as f, tqdm(
        total=total_length, unit="iB", unit_scale=True
    ) as progress:
        for chunk in resp.iter_content(chunk_size=1024):
            progress.update(len(chunk))
            f.write(chunk)
    return archive_file
