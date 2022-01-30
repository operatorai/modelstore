#    Copyright 2021 Neal Lathia
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

import click

from modelstore.utils import cli as modelstorecli


@click.group()
def download_model():
    pass


@click.group()
def upload_model():
    pass


@download_model.command()
@click.argument("domain", type=str, required=True)
@click.argument("model_id", type=str, required=True)
@click.argument("parent_dir", type=str, required=False, default=None)
def download(domain: str, model_id: str, parent_dir: str):
    """Download a model from the modelstore. Usage:\n
    ❯ python -m modelstore download <domain> <model-id> <parent-directory>
    """
    try:
        target_dir = (
            os.path.join(parent_dir, domain, model_id)
            if parent_dir is not None
            else os.path.join(domain, model_id)
        )
        os.makedirs(target_dir, exist_ok=True)

        model_store = modelstorecli.model_store_from_env()
        archive_path = model_store.download(target_dir, domain, model_id)
        modelstorecli.success(
            f"✅  Downloaded: {domain}={model_id} to {archive_path}"
        )
    except SystemExit:
        # Failed to instantiate a model store from environment variables
        pass
    except Exception:
        modelstorecli.failure("❌  Failed to download model:")
        raise


@upload_model.command()
@click.argument("domain", type=str, required=True)
@click.argument("model", type=click.Path(exists=True))
def upload(domain: str, model: str):
    """Upload a model to the modelstore. Usage:\n
    ❯ python -m modelstore upload <domain> /path/to/file
    """
    try:
        model_store = modelstorecli.model_store_from_env()
        meta_data = model_store.upload(domain, model=model)
        model_id = meta_data["model"]["model_id"]
        modelstorecli.success(f"✅  Uploaded: {domain}={model_id}")
        print(model_id)
    except SystemExit:
        # Failed to instantiate a model store from environment variables
        pass
    except Exception:
        modelstorecli.failure("❌  Failed to upload model:")
        raise


cli = click.CommandCollection(sources=[download_model, upload_model])

if __name__ == "__main__":
    cli()
