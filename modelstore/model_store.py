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
import os
import tarfile
from dataclasses import dataclass
from typing import Optional

from modelstore.models.managers import iter_libraries
from modelstore.storage.aws import BOTO_EXISTS, AWSStorage
from modelstore.storage.gcloud import GCLOUD_EXISTS, GoogleCloudStorage
from modelstore.storage.hosted import HostedStorage
from modelstore.storage.local import FileSystemStorage
from modelstore.storage.storage import CloudStorage


@dataclass(frozen=True)
class ModelStore:

    """
    ModelStore provides an class to:
    - Create archives containing trained models;
    - Upload those archives into cloud storage;
    - List and get info about the models you have stored in the cloud
    """

    # The backend provider, e.g. "gcloud"
    storage: CloudStorage

    @classmethod
    def from_aws_s3(cls, bucket_name: str, region: str = None) -> "ModelStore":
        """Creates a ModelStore instance that stores models to an AWS s3
        bucket.

        This currently assumes that the s3 bucket already exists."""
        if not BOTO_EXISTS:
            raise ModuleNotFoundError("boto3 is not installed!")
        return ModelStore(
            storage=AWSStorage(bucket_name=bucket_name, region=region)
        )

    @classmethod
    def from_gcloud(cls, project_name: str, bucket_name: str) -> "ModelStore":
        """Creates a ModelStore instance that stores models to a
        Google Cloud Bucket. This assumes that the Cloud bucket already exists."""
        if not GCLOUD_EXISTS:
            raise ModuleNotFoundError("google.cloud is not installed!")
        return ModelStore(
            storage=GoogleCloudStorage(project_name, bucket_name),
        )

    @classmethod
    def from_file_system(cls, root_directory: str) -> "ModelStore":
        """Creates a ModelStore instance that stores models to
        the local file system."""
        return ModelStore(storage=FileSystemStorage(root_directory))

    @classmethod
    def from_api_key(
        cls,
        access_key_id: Optional[str] = None,
        secret_access_key: Optional[str] = None,
    ) -> "ModelStore":
        """Creates a ModelStore instance that stores models to
        a managed system. Requires API keys."""
        return ModelStore(
            storage=HostedStorage(access_key_id, secret_access_key)
        )

    def __post_init__(self):
        if not self.storage.validate():
            raise Exception(
                f"Failed to set up the {type(self.storage).__name__} storage."
            )
        # Supported machine learning model libraries
        for library, manager in iter_libraries(self.storage):
            object.__setattr__(self, library, manager)

    def list_domains(self) -> list:
        """Returns a list of dicts, containing info about all
        of the domains"""
        return self.storage.list_domains()

    def list_versions(self, domain: str) -> list:
        """Returns a list of dicts, containing info about all
        of the models that have been uploaded to a domain"""
        return self.storage.list_versions(domain)

    def download(
        self, local_path: str, domain: str, model_id: str = None
    ) -> str:
        local_path = os.path.abspath(local_path)
        archive_path = self.storage.download(local_path, domain, model_id)
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(local_path)
        os.remove(archive_path)
        return local_path
