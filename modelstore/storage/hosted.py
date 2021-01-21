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
import threading
import urllib

import requests
from modelstore.storage.storage import CloudStorage
from modelstore.utils.log import logger
from tqdm import tqdm
from tqdm.utils import CallbackIOWrapper

# pylint: disable=protected-access

_URL_ROOT = "https://maamxzc32j.execute-api.eu-west-1.amazonaws.com/prod/"


class HostedStorage(CloudStorage):

    """
    HostedStorage is a managed model store. No dependencies required.
    Usage of this storage requires you to have an `api_key`.
    """

    def __init__(self, access_key_id: str, secret_access_key: str):
        super().__init__([])
        self.access_key_id = access_key_id
        self.secret_access_key = secret_access_key

    def validate(self) -> bool:
        """ Requires an ACCESS_KEY_ID and SECRET_ACCESS_KEY """
        for key in [self.access_key_id, self.secret_access_key]:
            if key is None or len(key) == 0:
                return False
        try:
            _ = self._post("ping", {})
            return True
        except:
            return False

    def _post(self, endpoint: str, data: dict) -> dict:
        logger.info(endpoint)
        url = os.path.join(_URL_ROOT, endpoint)
        headers = {"x-api-key": self.secret_access_key}
        data["api_key_id"] = self.access_key_id
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
        pass

    def upload(self, domain: str, model_id: str, local_path: str) -> dict:
        upload_url = self._get_url("upload-url", domain, model_id)
        _upload(local_path, upload_url)
        self._register_upload(domain, model_id)
        return {
            "type": "operator:cloud-storage",
        }

    def list_versions(self, domain: str) -> list:
        """ Returns a list of a model's versions """
        raise NotImplementedError()

    def list_domains(self) -> list:
        """ Returns a list of all the existing model domains """
        raise NotImplementedError()

    def set_meta_data(self, domain: str, model_id: str, meta_data: dict):
        """ Annotates a model with some given meta data """
        return

    def download(self, local_path: str, domain: str, model_id: str = None):
        """Downloads an artifacts archive for a given (domain, model_id) pair.
        If no model_id is given, it defaults to the latest model in that
        domain"""
        download_url = self._get_url("download-url", domain, model_id)
        return _download(local_path, download_url)


def _upload(local_path, upload_url):
    """ Uploads a local file to an upload URL, showing a progress bar """
    file_path = os.path.abspath(local_path)
    file_size = os.stat(file_path).st_size
    with open(file_path, "rb") as f:
        with tqdm(
            total=file_size, unit="B", unit_scale=True, unit_divisor=1024
        ) as t:
            wrapped_file = CallbackIOWrapper(t.update, f, "read")
            requests.put(upload_url, data=wrapped_file)


def _download(local_path, download_url) -> str:
    """ Uploads a local file to an upload URL, showing a progress bar """
    resp = requests.get(download_url, stream=True)
    total_length = int(resp.headers.get("content-length", 0))
    logger.info(total_length)
    archive_file = os.path.join(local_path, "artifacts.tar.gz")
    with open(archive_file, "wb") as f, tqdm(
        total=total_length, unit="iB", unit_scale=True
    ) as t:
        for chunk in resp.iter_content(chunk_size=1024):
            t.update(len(chunk))
            f.write(chunk)
    return archive_file
