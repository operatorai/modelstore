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
from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime

import modelstore
from modelstore.clouds.aws import BOTO_EXISTS, AWSStorage
from modelstore.clouds.file_system import FileSystemStorage
from modelstore.clouds.gcloud import GCLOUD_EXISTS, GoogleCloudStorage
from modelstore.clouds.storage import CloudStorage
from modelstore.meta import dependencies, revision, runtime
from modelstore.models.managers import ML_LIBRARIES, get_manager


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
    def from_aws_s3(cls, bucket_name: str, region: str = None) -> ModelStore:
        """Creates a ModelStore instance that stores models to an AWS s3
        bucket.
        
        This currently assumes that the s3 bucket already exists."""
        if not BOTO_EXISTS:
            raise ModuleNotFoundError("boto3 is not installed!")
        return ModelStore(
            storage=AWSStorage(bucket_name=bucket_name, region=region)
        )

    @classmethod
    def from_gcloud(cls, project_name: str, bucket_name: str) -> ModelStore:
        """Creates a ModelStore instance that stores models to a
        Google Cloud Bucket.
        
        This currently assumes that the Cloud bucket already exists."""
        if not GCLOUD_EXISTS:
            raise ModuleNotFoundError("google.cloud is not installed!")
        return ModelStore(
            storage=GoogleCloudStorage(project_name, bucket_name),
        )

    @classmethod
    def from_file_system(cls, root_directory: str) -> ModelStore:
        """Creates a ModelStore instance that stores models to
        the local file system. """
        return ModelStore(storage=FileSystemStorage(root_directory))

    def __post_init__(self):
        if not self.storage.validate():
            raise Exception(
                f"Failed to set up the {self.storage.name} storage."
            )
        # Supported machine learning model libraries
        for library in ML_LIBRARIES:
            object.__setattr__(self, library, get_manager(library)())

    def upload(self, domain: str, archive_path: str) -> dict:
        """Upload an archive to cloud storage. This function returns
        a dictionary of meta-data that is associated with this model,
        including an id.
        """
        _validate_domain(domain)
        model_id = str(uuid.uuid4())
        upload_time = datetime.now().strftime("%Y/%m/%d/%H:%M:%S")
        location = self.storage.upload(domain, upload_time, archive_path)

        meta_data = {
            "model": {"domain": domain, "model_id": model_id,},
            "storage": {"name": self.storage.get_name(), "location": location,},
            "meta": {
                "runtime": f"python:{runtime.get_python_version()}",
                "user": runtime.get_user(),
                "created": upload_time,
                "dependencies": dependencies.extract_dependencies(archive_path),
            },
            "modelstore": modelstore.__version__,
        }

        git_meta = revision.git_meta()
        if git_meta is not None:
            meta_data["meta"]["git"] = git_meta

        self.storage.set_meta_data(domain, model_id, meta_data)
        return meta_data

    def list_versions(self, domain: str) -> list:
        """ Returns a list of dicts, containing info about all
        of the models that have been uploaded to a domain """
        return self.storage.list_versions(domain)


def _validate_domain(domain: str):
    if len(domain) == 0:
        raise ValueError("Please provide a non-empty domain name.")
    if domain in [
        "versions",
        "domains",
        "modelstore",
        "operatorai-model-store",
    ]:
        raise ValueError("Please use a different domain name.")
