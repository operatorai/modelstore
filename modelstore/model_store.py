#    Copyright 2022 Neal Lathia
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
import tempfile
import warnings
from dataclasses import dataclass, asdict
from typing import Optional

from modelstore.ids import model_ids
from modelstore.storage.states.model_states import ReservedModelStates
from modelstore.models.managers import iter_libraries, matching_managers, get_manager
from modelstore.models.multiple_models import MultipleModelsManager
from modelstore.storage.aws import BOTO_EXISTS, AWSStorage
from modelstore.storage.azure import AZURE_EXISTS, AzureBlobStorage
from modelstore.storage.gcloud import GCLOUD_EXISTS, GoogleCloudStorage
from modelstore.storage.hdfs import HDFS_EXISTS, HdfsStorage
from modelstore.storage.local import FileSystemStorage
from modelstore.storage.minio import MINIO_EXISTS, MinIOStorage
from modelstore.storage.storage import CloudStorage
from modelstore.utils.exceptions import (
    ModelExistsException,
    DomainNotFoundException,
    ModelNotFoundException,
    ModelDeletedException,
)


@dataclass(frozen=True)
class ModelStore:

    """ModelStore is the main object that encapsulates a
    model registry. To create a new model store, use one of the
    ModelStore.from_ functions"""

    # The backend provider, e.g. "gcloud"
    storage: CloudStorage

    @classmethod
    def from_aws_s3(
        cls,
        bucket_name: Optional[str] = None,
        region: Optional[str] = None,
        root_prefix: Optional[str] = None,
    ) -> "ModelStore":
        """Creates a ModelStore instance that stores models to an AWS s3
        bucket.

        This currently assumes that the s3 bucket already exists."""
        if not BOTO_EXISTS:
            raise ModuleNotFoundError("boto3 is not installed!")
        return ModelStore(
            storage=AWSStorage(
                bucket_name=bucket_name, region=region, root_prefix=root_prefix
            )
        )

    @classmethod
    def from_azure(
        cls, container_name: Optional[str] = None, root_prefix: Optional[str] = None
    ) -> "ModelStore":
        """Creates a ModelStore instance that stores models to an
        Azure blob container. This assumes that the container
        already exists."""
        if not AZURE_EXISTS:
            raise ModuleNotFoundError("azure-storage-blob is not installed!")
        return ModelStore(
            storage=AzureBlobStorage(
                container_name=container_name, root_prefix=root_prefix
            )
        )

    @classmethod
    def from_gcloud(
        cls,
        project_name: Optional[str] = None,
        bucket_name: Optional[str] = None,
        root_prefix: Optional[str] = None,
    ) -> "ModelStore":
        """Creates a ModelStore instance that stores models to a
        Google Cloud Bucket. This assumes that the Cloud bucket
        already exists."""
        if not GCLOUD_EXISTS:
            raise ModuleNotFoundError("google.cloud is not installed!")
        return ModelStore(
            storage=GoogleCloudStorage(
                project_name, bucket_name, root_prefix=root_prefix
            )
        )

    @classmethod
    def from_minio(
        cls,
        endpoint: Optional[str] = None,  # Defaults to s3.amazonaws.com
        access_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        bucket_name: Optional[str] = None,
        root_prefix: Optional[str] = None,
        secure: Optional[bool] = True,
    ) -> "ModelStore":
        """Creates a ModelStore instance that stores models using a MinIO client.
        This assumes that the bucket already exists."""
        if not MINIO_EXISTS:
            raise ModuleNotFoundError("minio is not installed!")
        return ModelStore(
            storage=MinIOStorage(
                endpoint=endpoint,
                access_key=access_key,
                secret_key=secret_key,
                bucket_name=bucket_name,
                root_prefix=root_prefix,
                secure=secure,
            )
        )

    @classmethod
    def from_hdfs(
        cls, root_prefix: Optional[str] = None, create_directory: bool = False
    ) -> "ModelStore":
        """Creates a ModelStore instance that stores models to
        the local HDFS system."""
        if not HDFS_EXISTS:
            raise ModuleNotFoundError("pydoop is not installed!")
        return ModelStore(storage=HdfsStorage(root_prefix, create_directory))

    @classmethod
    def from_file_system(
        cls, root_directory: Optional[str] = None, create_directory: bool = False
    ) -> "ModelStore":
        """Creates a ModelStore instance that stores models to
        the local file system."""
        return ModelStore(storage=FileSystemStorage(root_directory, create_directory))

    def __post_init__(self):
        if not self.storage.validate():
            raise Exception(
                f"Failed to set up the {type(self.storage).__name__} storage."
            )
        # Add attributes for ML libraries that exist in the current
        # environment
        libraries = []
        for library, manager in iter_libraries(self.storage):
            object.__setattr__(self, library, manager)
            libraries.append(manager)
        object.__setattr__(self, "_libraries", libraries)

    # pylint: disable=pointless-string-statement
    """
    DOMAINS: a domain is a string that is used to group several models together
    (e.g., belonging to the same end usage). Domains are created automatically
    when a model is first uploaded into it.
    """

    def list_domains(self) -> list:
        """Returns a list of dicts, containing info about all
        of the domains"""
        return self.storage.list_domains()

    def get_domain(self, domain: str) -> dict:
        """Returns the meta-data about a domain"""
        return self.storage.get_domain(domain)

    """
    MODELS: multiple models can be added to a domain;
    """

    def list_versions(self, domain: str, state_name: Optional[str] = None) -> list:
        """Lists the models in a domain (deprecated)"""
        warnings.warn(
            "list_versions() is deprecated; use list_models()",
            category=DeprecationWarning,
        )
        return self.list_models(domain, state_name)

    def list_models(self, domain: str, state_name: Optional[str] = None) -> list:
        """Returns a list of dicts, containing info about the
        models that have been uploaded to a domain; if state_name
        is given results are filtered to models set to that state"""
        return self.storage.list_models(domain, state_name)

    """
    MODEL STATES: a model state is a string that has a 1:many relationship
    with models.

    @TODO: There is no function to get the meta-data for a state
    @TODO: Model states are currently raw dictionaries
    """

    def list_model_states(self) -> list:
        """Returns a list of the available model states that
        have been created with `create_model_state()`"""
        return self.storage.list_model_states()

    def create_model_state(self, state_name: str):
        """Creates a state label models (e.g., shadow/prod/archived).
        There are some values that are reserved, see modelstore/storage/states/model_states.py
        """
        return self.storage.create_model_state(state_name)

    def set_model_state(self, domain: str, model_id: str, state_name: str):
        """Sets the model_id model to a specific state.
        That state must already exist (ref: `create_model_state()`) unless
        it is a reserved value (modelstore/storage/states/model_states.py)
        """
        return self.storage.set_model_state(domain, model_id, state_name)

    def remove_model_state(self, domain: str, model_id: str, state_name: str):
        """Removes a model_id from a specific state.
        This will not error if the model was never set to that state to begin
        with, but it will if that state does not exist"""
        return self.storage.unset_model_state(domain, model_id, state_name)

    """
    MODELS: a model archive is created and stored when using upload(). Each model
    is given an id, which can be used to retrieve meta data about it. Models
    can be downloaded as files, or loaded straight back into memory.
    """

    def get_model_info(self, domain: str, model_id: str) -> dict:
        """Returns the meta-data for a given model"""
        return asdict(self.storage.get_meta_data(domain, model_id))

    def model_exists(self, domain: str, model_id: str) -> bool:
        """Returns True if a model with the given id exists in the domain"""
        try:
            self.storage.get_meta_data(domain, model_id)
            return True
        except DomainNotFoundException:
            # The domain does not exist, so the model
            # does not exist either
            return False
        except ModelNotFoundException:
            # The domain exists, but the model_id in that
            # domain does not
            return False

    def upload(self, domain: str, model_id: Optional[str] = None, **kwargs) -> dict:
        """Creates an archive for a model (from the kwargs), uploads it
        to storage, and returns a dictionary of meta-data about the model"""
        # Generate an ID and validate it -- if no model id is given, then modelstore
        # defaults to using a uuid4 ID.
        model_id = str(model_id) if model_id is not None else model_ids.new()
        if not model_ids.validate(model_id):
            raise ValueError(f"model_id='{model_id}' contains invalid characters")

        # Figure out which library the kwargs match with
        # We do this _before_ checking whether the model exists to raise
        # catch if the kwargs aren't quite right before potentially modifying
        # model state, below
        # pylint: disable=no-member
        managers = matching_managers(self._libraries, **kwargs)
        if len(managers) == 1:
            manager = managers[0]
        else:
            # There are cases where we can match on more than one
            # manager (e.g., a model and an explainer)
            manager = MultipleModelsManager(managers, self.storage)

        try:
            if self.model_exists(domain, model_id):
                raise ModelExistsException(domain, model_id)
        except ModelDeletedException:
            # If a model has been deleted, then it _technically_ does not
            # exist anymore and we allow a new model to replace it. To
            # ensure that meta-data remains consistent, we remove the model
            # from the 'deleted' state here before uploading the new model
            self.storage.unset_model_state(
                domain,
                model_id,
                ReservedModelStates.DELETED.value,
                modifying_reserved=True,
            )
        meta_data = manager.upload(domain, model_id=model_id, **kwargs)
        return asdict(meta_data)

    def load(self, domain: str, model_id: str):
        """Loads a model into memory"""
        meta_data = self.storage.get_meta_data(domain, model_id)
        model_type = meta_data.model_type()
        if model_type.library == MultipleModelsManager.NAME:
            manager = MultipleModelsManager([], self.storage)
        else:
            manager = get_manager(model_type.library, self.storage)
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_files = self.download(tmp_dir, domain, model_id)
            return manager.load(model_files, meta_data)

    def download(self, local_path: str, domain: str, model_id: str = None) -> str:
        """Downloads the model a domain to local_path"""
        local_path = os.path.abspath(local_path)
        archive_path = self.storage.download(local_path, domain, model_id)
        with tarfile.open(archive_path, "r:gz") as tar:

            def is_within_directory(directory, target):
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
                prefix = os.path.commonprefix([abs_directory, abs_target])
                return prefix == abs_directory

            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
                tar.extractall(path, members, numeric_owner=numeric_owner)

            safe_extract(tar, local_path)
        os.remove(archive_path)
        return local_path

    def delete_model(self, domain: str, model_id: str, skip_prompt: bool = False):
        """Deletes a model artifact from storage."""
        meta_data = self.storage.get_meta_data(domain, model_id)
        self.storage.delete_model(domain, model_id, meta_data, skip_prompt)
