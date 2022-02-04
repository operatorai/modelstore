import os

from modelstore import ModelStore


def create_model_store(backend) -> ModelStore:
    modelstores = {
        "aws": create_aws_model_store,
        "azure": create_azure_model_store,
        "gcloud": create_gcloud_model_store,
        "filesystem": create_file_system_model_store,
    }
    return modelstores[backend]()


def create_aws_model_store() -> ModelStore:
    # A model store in an AWS S3 bucket
    # The modelstore library assumes you have already created
    # an s3 bucket and will raise an exception if it doesn't exist
    return ModelStore.from_aws_s3(
        os.environ["MODEL_STORE_AWS_BUCKET"],
        root_prefix="example-by-ml-library",
    )


def create_azure_model_store() -> ModelStore:
    # A model store in an Azure Container
    # The modelstore library assumes that:
    # 1. You have already created an Azure container
    # 2. You have an os environment variable called AZURE_STORAGE_CONNECTION_STRING
    return ModelStore.from_azure(
        container_name=os.environ["MODEL_STORE_AZURE_CONTAINER"],
    )


def create_gcloud_model_store() -> ModelStore:
    # A model store in a Google Cloud Bucket
    # The modelstore library assumes you have already created
    # a Cloud Storage bucket and will raise an exception if it doesn't exist
    return ModelStore.from_gcloud(
        os.environ["MODEL_STORE_GCP_PROJECT"],
        os.environ["MODEL_STORE_GCP_BUCKET"],
    )


def create_file_system_model_store() -> ModelStore:
    # A model store in a local file system
    # Here, we create a local model store the location specified
    # in the MODEL_STORE_ROOT environment variable; and default to the
    # user's home directory if that environment variable does not exist
    root_dir = os.environ.get("MODEL_STORE_ROOT", os.path.expanduser("~"))
    print(f"🏦  Creating store in: {root_dir}")
    return ModelStore.from_file_system(root_directory=root_dir)
