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

import requests
from modelstore.storage.storage import CloudStorage
from modelstore.utils.log import logger
from tqdm import tqdm
from tqdm.utils import CallbackIOWrapper

# pylint: disable=protected-access

_URL_ROOT = "https://europe-west1-revival-287212.cloudfunctions.net/"


class HostedStorage(CloudStorage):

    """
    HostedStorage is a managed model store. No dependencies required.
    Usage of this storage requires you to have an `api_key`.
    """

    def __init__(self, api_key: str):
        super().__init__([])
        self.api_key = api_key

    def validate(self) -> bool:
        """ No dependencies or setup required; validation returns True """
        return self.api_key is not None and len(self.api_key) > 0

    def _get_url(self, domain: str, model_id: str, url_type: str) -> str:
        """ Returns a pre-signed URL for up/downloading models """
        endpoint = os.path.join(_URL_ROOT, "model-generate-url")
        body = {
            "api_key": self.api_key,
            "domain": domain,
            "model_id": model_id,
            "url_type": url_type,
        }
        rsp = requests.post(endpoint, data=json.dumps(body))
        if rsp.status_code == 200:
            return rsp.json()["url"]
        raise Exception(f"Error: {rsp.text.strip()}")

    def _upload(self, local_path, upload_url):
        """ Uploads a local file to an upload URL, showing a progress bar """
        file_path = os.path.abspath(local_path)
        file_size = os.stat(file_path).st_size
        with open(file_path, "rb") as f:
            with tqdm(
                total=file_size, unit="B", unit_scale=True, unit_divisor=1024
            ) as t:
                wrapped_file = CallbackIOWrapper(t.update, f, "read")
                requests.put(upload_url, data=wrapped_file)

    def _register_upload(self, domain: str, model_id: str):
        endpoint = os.path.join(_URL_ROOT, "model-register-uploaded")
        body = {
            "domain": domain,
            "model_id": model_id,
            "api_key": self.api_key,
        }
        rsp = requests.post(endpoint, data=json.dumps(body))
        if rsp.status_code != 200:
            raise Exception(f"Error: {rsp.text.strip()}")

    def upload(self, domain: str, model_id: str, local_path: str) -> dict:
        upload_url = self._get_url(domain, model_id, "PUT")
        self._upload(local_path, upload_url)
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
        # @TODO
        return

    def download(self, local_path: str, domain: str, model_id: str = None):
        """Downloads an artifacts archive for a given (domain, model_id) pair.
        If no model_id is given, it defaults to the latest model in that
        domain"""
        raise NotImplementedError()
