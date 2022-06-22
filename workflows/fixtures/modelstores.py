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

from modelstore import ModelStore


def create_model_store(backend) -> ModelStore:
    """ Returns a modelstore instance with the required storage type """
    modelstores = {
        "aws": create_aws_model_store,
        "azure": create_azure_model_store,
        "gcloud": create_gcloud_model_store,
        "filesystem": create_file_system_model_store,
    }
    return modelstores[backend]()


def create_aws_model_store() -> ModelStore:
    """ A model store that uses an s3 bucket """
    return ModelStore.from_aws_s3(
        os.environ["MODEL_STORE_AWS_BUCKET"],
        root_prefix="example-by-ml-library",
    )


def create_azure_model_store() -> ModelStore:
    """ A model store that uses an azure container """
    # A model store in an Azure Container
    # The modelstore library assumes that:
    # 1. You have already created an Azure container
    # 2. You have an os environment variable called AZURE_STORAGE_CONNECTION_STRING
    return ModelStore.from_azure(
        container_name=os.environ["MODEL_STORE_AZURE_CONTAINER"],
    )


def create_gcloud_model_store() -> ModelStore:
    """ A model store in a Google Cloud bucket """
    # The modelstore library assumes you have already created
    # a Cloud Storage bucket and will raise an exception if it doesn't exist
    return ModelStore.from_gcloud(
        os.environ["MODEL_STORE_GCP_PROJECT"],
        os.environ["MODEL_STORE_GCP_BUCKET"],
    )


def create_file_system_model_store() -> ModelStore:
    """ A model store in a local file system """
    # Here, we create a new local model store in our home directory
    home_dir = os.path.expanduser("~")
    print(f"🏦  Creating store in: {home_dir}")
    return ModelStore.from_file_system(root_directory=home_dir)
