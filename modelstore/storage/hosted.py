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
from modelstore.storage.storage import CloudStorage
from modelstore.utils.log import logger

# pylint: disable=protected-access

_URL_ROOT = "https://europe-west1-revival-287212.cloudfunctions.net/"


class HostedStorage(CloudStorage):

    """
    OperatorStorage is a managed model store. No dependencies required.

    Usage of this storage requires you to have an `api_key`.
    """

    def __init__(self, api_key: str):
        super().__init__([])
        self.api_key = api_key

    def validate(self) -> bool:
        """ No dependencies or setup required; validation returns True """
        return self.api_key is not None and len(self.api_key) > 0

    # def get_url(self, )

    # def _push(self, source: str, destination: str):
    # @TODO use curl to upload the model
    # logger.debug("Uploading from: %s...", source)

    def upload(self, domain: str, model_id: str, local_path: str) -> dict:
        # @TODO request a presigned URL
        # @TODO self._push()
        # @TODO register upload
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
