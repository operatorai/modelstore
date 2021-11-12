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
import click

from modelstore.storage.util.environment import model_store_from_env
from modelstore.utils import cli as modelstorecli


@click.group()
def download_model():
    pass


@download_model.command()
@click.argument("domain", type=str, required=True)
@click.argument("model_id", type=str, required=True)
@click.argument("target_dir", type=str, required=False, default=None)
def download(domain, model_id, target_dir):
    """Download a model from the modelstore. Usage:\n
    ❯ python -m modelstore <domain> <model-id> <target-directory>
    """
    try:
        model_store = model_store_from_env()
        model_store.download(target_dir, domain, model_id)
        modelstorecli.success(
            f"✅  Downloaded: {domain}={model_id} to {target_dir}"
        )
    except:
        modelstorecli.failure("❌  Failed to download model:")
        raise


cli = click.CommandCollection(sources=[download_model])

if __name__ == "__main__":
    cli()
