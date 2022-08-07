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
import tempfile
import click

from actions import actions
from fixtures import extra
from fixtures.models import iter_models, iter_model_files
from fixtures.modelstores import create_model_store


@click.command()
@click.option(
    "--modelstore-in",
    type=click.Choice(["aws", "azure", "gcloud", "filesystem"]),
)
def main(modelstore_in: str):
    """ Executes all of the integration tests in a given storage type """
    print(f"🆕  Running modelstore example with {modelstore_in} backend.")

    # Create a modelstore instance
    model_store = create_model_store(modelstore_in)

    # Run actions on the modelstore instance that are
    # not dependent on a trained model (e.g. model states)
    actions.run_on_storage(model_store)

    # Run actions on the modelstore instance that
    # are dependent on a trained model (e.g. upload, download)
    with tempfile.TemporaryDirectory() as tmp_dir:
        extra_files = extra.files(tmp_dir, num_files=2)
        for model in iter_models():
            actions.run_with_model(
                model_store,
                model,
                extra_metadata=extra.metadata(),
                extra_files=extra_files,
            )

    # Run CLI actions
    with tempfile.TemporaryDirectory() as tmp_dir:
        for model_path in iter_model_files(tmp_dir):
            actions.run_cli_commands(model_path)


if __name__ == "__main__":
    main()
